import os
import sys

# Inject SUMO_HOME
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'

import sumo_rl.environment.env as env_mod
import traci

# Intercept traci.start to print exactly what is passed to SUMO
old_traci_start = traci.start
def logging_traci_start(cmd, *args, **kwargs):
    print("========================================")
    print("STARTING TRACI/SUMO WITH COMMAND:")
    print(" ".join(cmd))
    print("========================================")
    return old_traci_start(cmd, *args, **kwargs)

traci.start = logging_traci_start

try:
    env = env_mod.SumoEnvironment(
        net_file='data/bremen.net.xml',
        route_file='data/bremen.rou.xml',
        additional_sumo_cmd='-a data/bremen.add.xml'
    )
    env.reset()
except Exception as e:
    print(f"Exception caught: {e}")
