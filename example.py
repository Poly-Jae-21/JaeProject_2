from model.build import argument
from env.MultiCity.Chicago.chicago import ChicagoEnv

config = argument()

env = ChicagoEnv(config)
a = env.reset()