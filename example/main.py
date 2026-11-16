import nematocyte

from src.run import Config

# This is the main entry point for the script.
# Meaning of the arguments to @nematocyte.main():
#     Config: The dataclass that defines which keys and values are valid in the
#         configuration. The function below receives an instance of it.
#     config_dir: The path to the directory containing the yaml config files,
#         relative to this file.
# There is no default config file: if you run the script without `--config-name`,
# all values come from the defaults in the `Config` class.


@nematocyte.main(Config, config_dir="configs")
def main(config: Config) -> float:
    # `to_dict` turns the config object into plain dicts, lists, strings and numbers,
    # which is what we want to send to the logger.
    config_for_logging = nematocyte.to_dict(config)

    # Finally, we call the `run` method of the `Config` object.
    return config.run(config_for_logging)


if __name__ == "__main__":
    main()
