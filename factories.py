from base import SequentialModule, StatelessWrapper#, NewNetworkModule
from torch import nn

def flatten_to_int(shape):
    out = 1
    for s in shape:
        out *= s
    return out

def flatten_shape(shape):
    if shape is None:
        return None
    out = 1
    for s in shape:
        out *= s
    return (out,)

class ModuleFactory:
    def __init__(self):
        self.module_class = None
        self.args = None

    def shape_change(self, in_shape):
        return in_shape

    def make_module(self, in_shape):
        return self.module_class(in_shape, *self.args)

    def assemble_module(self, in_shape, unrolled):
        return self.make_module(in_shape)

    def __call__(self, in_size):
        return self.module_class(in_size, *self.args)


class SimpleDataFactory(ModuleFactory):
    def assemble_module(self, in_shape, unrolled):
        return self.make_module(flatten_shape(in_shape))


class Linear(SimpleDataFactory):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size

    def shape_change(self, in_shape):
        return (self.out_size,)

    def make_module(self, in_shape):
        return StatelessWrapper(in_shape, nn.Linear(in_shape[0], self.out_size), (self.out_size,))



class Sequential(ModuleFactory):
    def __init__(self, *module_facs):
        super().__init__()
        self.module_facs = module_facs


    def shape_change(self, in_shape):
        cur_shape = in_shape
        for factory in self.module_facs:
            cur_shape = factory.size_change(cur_shape)
        return cur_shape

    def __call__(self, inp_info):
        child_info = {**inp_info}
        mlist = []
        for factory in self.module_facs:
            new_module = factory(child_info)
            mlist.append(new_module)
            child_info['in_shape'] = new_module.get_out_shape()
        return SequentialModule(inp_info['in_shape'], mlist)


class ExecPath:
    def __init__(self, inputs, module_factories, result_name):
        self.name = result_name
        self.inputs = inputs
        self.factory = module_factories

    '''
    def deduce_size(self, other_sizes):
        if all([x in other_sizes for x in self.inputs]):
            cur_size = other_sizes[self.inputs[0]] if len(self.inputs) == 1 else torch.tensor([other_sizes[x] for x in self.inputs]).sum().item()
        else:
            cur_size = 0
        for factory in self.strand:
            #print(cur_size)
            cur_size = factory.size_change(cur_size)
        if not isinstance(cur_size, int):
            cur_size = flatten(cur_size)
        if cur_size <= 0:
            raise Exception('Size deduction failed.')
        return cur_size
    '''

class Network(ModuleFactory):
    def __init__(self, *args):
        super().__init__()
        self.paths = args
        self.module_factories = {path.name: path.factory for path in self.paths}
        self.inputs = {path.name: path.inputs for path in self.paths}
        self.og_order = [path.name for path in self.paths]


    def _compute_shapes(self, in_shape):
        shapes = {'input': in_shape, **{layer: None for layer in self.og_order}}
        paths = set(self.paths)
        while paths:
            found_new = None
            for path in paths:
                inp_shape = shapes[path.name] if len(path.inputs) == 1 else flatten_shape(shapes[path.name])
                shapes[path.name] = path.factory.shape_change(inp_shape)
                if shapes[path.name] is not None:
                    found_new = path
                    break
            paths.remove(found_new)
        return shapes

    def shape_change(self, in_shape):
        shapes = self._compute_shapes(in_shape)
        return shapes['output']

    def _compute_execution_order(self):
        order = []
        inputs = {'input': []}
        reachable = {}
        for path in self.paths:
            order.append(path.name)
            inputs[path.name] = path.inputs
        for path, inp_list in inputs.items():
            visited = {*inp_list}
            todo = {*inp_list}
            while todo:
                todo.update(set(inputs[todo.pop()]).difference(visited))
                visited.update(todo)
            reachable[path] = visited
        cycles_list = []
        in_cycles = set()
        for path, reach in reachable.items():
            if path in in_cycles:
                continue
            cycle_with = []
            for other in reach:
                if path in reachable[other]:
                    cycle_with.append(other)
            if cycle_with:
                cycle_with.sort(key=order.index)
                cycles_list.append(cycle_with)
                in_cycles.update(cycle_with)
        cycles_dict = {f'c{i}': cycle for i, cycle in enumerate(cycles_list)}
        requirements = {key: set(value) for key, value in inputs.items()}
        for c_name, cycle in cycles_dict.items():
            requirement = set()
            for path in cycle:
                requirement.update(inputs[path])
            requirements[c_name] = requirement.difference(cycle)
        nodes = set(order).difference(in_cycles).union(cycles_dict.keys())
        computed_order = []
        done = {'input'}
        while nodes:
            for node in nodes:
                if requirements[node].issubset(done):
                    done.add(node)
                    if node in cycles_dict:
                        done.update(cycles_dict[node])
                    computed_order.append(node)
            nodes.difference_update(done)
        return computed_order, cycles_dict

    def _build_inner_network(self, in_shape, order, shapes):
        module_dict = nn.ModuleDict()
        for layer in order:
            module_dict[layer] = self.module_factories[layer].assemble_module(shapes[layer], True)
        inputs = {layer: self.inputs[layer] for layer in order}
        seen = set()
        recurrent_layers = set()
        for layer in order:
            recurrent_layers.update(set(inputs[layer]).difference(seen))
            seen.add(layer)
        recurrent_layers.intersection_update(order)
        return NewNetworkModule(in_shape, order, inputs, module_dict, {}, recurrent_layers)

    def assemble_module(self, in_shape, unrolled):
        shapes = self._compute_shapes(in_shape)
        if not unrolled:
            module_dict = nn.ModuleDict()
            outer_order, outer_layers, cycles_layers, new_inputs, cycles_outputs = self._compute_execution_order()
            for name, cycle in cycles_layers:
                module_dict[name] = self._build_inner_network(in_shape, cycle, shapes)
            for name in outer_layers:
                module_dict[name] = self.module_factories[name].assemble_module(shapes[name], False)
            return NewNetworkModule(in_shape, outer_order, new_inputs, module_dict, cycles_outputs, [])
        else:
            return self._build_inner_network(in_shape, self.og_order, shapes)






        # wrap cycles in inner networks
        # deduce shapes