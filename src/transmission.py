class TransmissionSystem:
    def __init__(self, Sbase):
        self.Sbase = Sbase
        self.path = []  # Ordered list of components in the path

    def add_source(self, voltage, voltage_angle):
        self.path.append({"type": "source", "voltage": voltage, "voltage_angle": voltage_angle})
        return self 

    def add_impedance(self, impedance):
        self.path.append({"type": "impedance", "impedance": impedance})
        return self  

    def add_load(self, component_index, S):
       self.path.append({"type": "load", "component_index": component_index, "S": S})
       return self

    def build(self):
        self.line.build()
        self.xfmer.build()