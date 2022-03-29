def get_artifact_name_OLD(robot, is_small: bool, tag: str):
    """Return artifact name.
    regular dataset, no tag: <robot>
    regular dataset, tag: <robot>_<tag>
    small dataset, no tag: <robot>_SMALL
    small dataset, tag: <robot>_SMALL_<tag>
    """
    name = robot
    if is_small:
        name += "_SMALL"

    if tag is not None:
        if len(tag) > 0:
            name += "_" + tag
    return name


def get_artifact_name(robot: str, is_small: bool, tag: str):
    """Return artifact name.
    Per wandb: 'Artifact name may only contain alphanumeric characters, dashes, underscores, and dots.'
    """
    return f"robot-{robot}__is_small-{is_small}__tag-{tag}"
