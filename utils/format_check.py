import json


def check_output_format(output_str):
    """
    Check if the output format is correct.

    :param output_str: str, the JSON string to check
    :return: bool, str
    """
    try:
        output = json.loads(output_str)
    except json.JSONDecodeError:
        return False, "Output is not valid dict."

    if not isinstance(output, dict):
        return False, "Output should be a dictionary."

    # Check for required keys
    required_keys = {"action_list", "reason", "observation"}
    if not required_keys.issubset(output.keys()):
        return False, f"Output is missing required keys: {required_keys - output.keys()}"

    # Check action_list format
    if not isinstance(output["action_list"], list):
        return False, "action_list should be a list."

    for action in output["action_list"]:
        if not (isinstance(action, list) and len(action) == 2):
            return False, "Each action should be a list [action, duration]."
        if not isinstance(action[0], str):
            return False, "The action should be a string."
        if not isinstance(action[1], (int, float)):
            return False, "The duration should be a number."

    # Check reason and observation format
    if not isinstance(output["reason"], str):
        return False, "reason should be a string."

    if not isinstance(output["observation"], str):
        return False, "observation should be a string."

    return True, output

#
# # Example usage
# output_example_str = '{"action_list": [["MOVE_FORWARD", 0.5], ["MOVE_LEFT", 1]], "reason": "To explore the map.", "observation": "A clear path with no obstacles."}'
# is_correct, message = check_output_format(output_example_str)
# print(is_correct, message)