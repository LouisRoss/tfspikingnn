class InitLoader:
    def __init__(self, module_name, configuration):
        self.module = __import__(module_name)
        MakeInitializer = getattr(self.module, 'MakeInitializer')
        self.init_class = MakeInitializer(configuration)

    def InitializeConnections(self):
        return self.init_class.InitializeConnections()

    def GenerateSpikes(self, duration):
        return self.init_class.GenerateSpikes(duration)
