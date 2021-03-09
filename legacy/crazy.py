from factories import ModuleFactory, flatten_shape
from base import NewNetworkModule
from torch import nn

class LayerBase:
    pass

class Placeholder(LayerBase):
    inputs = set()

    def assign(self, layer):
        self.layer = layer

class Network(ModuleFactory):
    def __init__(self):
        super().__init__() # TODO: check setattr calls in this
        self._layers = {}
        self._laph = {}
        self._reverse_laph = {}
        self._og_order = []
        self.input = Placeholder()
        self._reverse_laph[self.input] = 'input'  # TODO: reverse?
        self._laph['input'] = self.input

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if not isinstance(value, LayerBase):
            return
        if key in self._laph:
            if isinstance(self._laph[key], Placeholder):
                self._laph[key].assign(value)
                if value.placeholder and value.placeholder is not self._laph[key]:
                    raise Exception('Two placeholders given for a layer.')
                value.placeholder = self._laph[key]
                self._reverse_laph[self._laph[key]] = key+'_ph'
                self._laph[key+'_ph'] = self._laph[key]
            else:
                raise Exception('Overwriting layers with other layers is not allowed.')
        if isinstance(value, Layer):
            for input in value.inputs:
                if input not in self._reverse_laph:
                    raise Exception('Input is not in network.')
            self._layers[key] = value
            self._og_order.append(key)
        self._laph[key] = value
        self._reverse_laph[value] = key


    def _compute_shapes(self, in_shape):
        shapes = {'input': in_shape, **{layer: None for layer in self._layers}}
        layers = set(self._layers)
        while layers:
            found_new = None
            for layer_name in layers:
                inp_shape = shapes[layer_name] if len(self._layers[layer_name].inputs) == 1 else flatten_shape(shapes[layer_name])
                shapes[layer_name] = self._layers[layer_name].factory.shape_change(inp_shape)
                if shapes[layer_name] is not None:
                    found_new = layer_name
                    break
            layers.remove(found_new)
        return shapes

    def shape_change(self, in_shape):
        shapes = self._compute_shapes(in_shape)
        return shapes['output']

    def _compute_execution_order(self, input_names):
        #order = []
        #inputs = {'input': []}
        #for path in self.paths:
        #    order.append(path.name)
        #    inputs[path.name] = path.inputs

        #Find reachables
        reachable = {self.input: set()}
        for layer_name, layer in self._layers.items():
            visited = {*layer.inputs}
            todo = {*layer.inputs}
            while todo:
                todo.update(set(todo.pop().inputs).difference(visited))
                visited.update(todo)
            reachable[layer] = visited


        cycles_list = []
        in_cycles = set()
        for layer, reach in reachable.items():
            if self._reverse_laph[layer] in in_cycles:
                continue
            cycle_with = []
            for other in reach:
                if layer in reachable[other]:
                    cycle_with.append(self._reverse_laph[other])
            if cycle_with:
                cycle_with.sort(key=self._og_order.index)
                cycles_list.append(cycle_with)
                in_cycles.update(cycle_with)
        cycles_dict = {f'c{i}': cycle for i, cycle in enumerate(cycles_list)}

        requirements = {**input_names}#{key: set(value.inputs) for key, value in self._layers.items()}
        for c_name, cycle in cycles_dict.items():
            req = set()
            for layer_name in cycle:
                req.update(input_names[layer_name])
            requirements[c_name] = req.difference(cycle)

        remaining = set(self._og_order).difference(in_cycles)
        nodes = remaining.union(cycles_dict.keys())
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

        req_remain = set.union(*{input_names[layer_name] for layer_name in remaining})
        cycle_outputs = {c_name: req_remain.intersection(cycle) for c_name, cycle in cycles_dict}

        return computed_order, remaining, cycles_dict, requirements, cycle_outputs

    def _build_inner_network(self, in_shape, order, shapes, input_names):
        module_dict = nn.ModuleDict()
        for layer in order:
            module_dict[layer] = self._layers[layer].factory.assemble_module(shapes[layer], True)
        inputs = {layer: input_names[layer] for layer in order}
        seen = set()
        recurrent_layers = set()
        for layer in order:
            recurrent_layers.update(set(inputs[layer]).difference(seen))
            seen.add(layer)
        recurrent_layers.intersection_update(order)
        return NewNetworkModule(in_shape, order, inputs, module_dict, {}, recurrent_layers)

    def assemble_module(self, in_shape, unrolled):
        shapes = self._compute_shapes(in_shape)
        input_names = {layer_name: {self._reverse_laph[inp_lay] for inp_lay in layer.inputs} for layer_name, layer in self._layers.items()}
        if not unrolled:
            module_dict = nn.ModuleDict()
            outer_order, outer_layers, cycles_layers, new_inputs, cycles_outputs = self._compute_execution_order(input_names)
            for name, cycle in cycles_layers:
                module_dict[name] = self._build_inner_network(in_shape, cycle, shapes, input_names)
            for name in outer_layers:
                module_dict[name] = self._layers[name].factory.assemble_module(shapes[name], False)
            return NewNetworkModule(in_shape, outer_order, new_inputs, module_dict, cycles_outputs, [])
        else:
            return self._build_inner_network(in_shape, self._og_order, shapes, input_names)




class Layer(LayerBase):
    def __init__(self, inputs, factory, placeholder=None):
        try:
            iter(inputs)
            self.inputs = inputs
        except TypeError:
            self.inputs = (inputs,)
        if not isinstance(factory, ModuleFactory):
            pass  # TODO: use Sequential
        else:
            self.factory = factory
        self.placeholder = placeholder

