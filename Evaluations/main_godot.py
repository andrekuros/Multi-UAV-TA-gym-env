

def createGodotEnv():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--env_path",
        default="W:/OneDrive/Doutorado/SwarmSimASA/GoDot/godot_rl_agents-main/examples/godot_rl_agents_examples-main/godot_rl_agents_examples-main/examples/FlyBy/bin/FlyBy.exe",#envs/example_envs/builds/JumperHard/jumper_hard.x86_64",        
        type=str,
        help="The Godot binary to use, do not include for in editor training",
    )
    
    parser.add_argument("--speedup", default=10, type=int, help="whether to speed up the physics in the env")
    parser.add_argument("--renderize", default=1, type=int, help="whether renderize or not the screen")
    args, extras = parser.parse_known_args()
    env = GodotEnv( env_path=args.env_path,
             port=11008,
             show_window=True,
             seed=0,
             framerate=None,
             action_repeat=60,
             speedup=args.speedup,
             convert_action_space=False,
             renderize=args.renderize
             )

    return env
