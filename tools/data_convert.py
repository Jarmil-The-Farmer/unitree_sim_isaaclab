def convert_to_joint_range(value):
    """Convert the command value to the Isaac Lab joint angle [5.6, 0] -> [-0.02, 0.024]
    
    Args:
        value: the input value, range in [5.6, 0]
                5.6: fully open
                0.0: fully closed
        
    Returns:
        float: the converted value, range in [-0.02, 0.03]
                -0.02: fully open
                0.03: fully closed
    """
    # input range (gripper control value)
    input_min = 0.0    # fully closed
    input_max = 5.4    # fully open
    
    # output range (joint angle)
    output_min = 0.024  # fully closed
    output_max = -0.02 # fully open
    
    # ensure the input value is in the valid range
    value = max(input_min, min(input_max, value))
    
    # linear mapping conversion
    converted_value = output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)
    
    return converted_value

def convert_to_gripper_range(value):
    """Convert the Isaac Lab joint angle to the gripper control value [-0.02, 0.03] -> [5.6, 0]
    
    Args:
        value: the input value, range in [-0.02, 0.024]
                -0.02: fully open
                0.03: fully closed
        
    Returns:
        float: the converted value, range in [5.6, 0]
                5.6: fully open
                0.0: fully closed
    """
    # input range (joint angle)
    input_min = 0.024   # fully closed
    input_max = -0.02  # fully open
    
    # output range (gripper control value)
    output_min = 0.0   # fully closed
    output_max = 5.4   # fully open
    try:
        value = round(float(value), 3)
    except Exception:
        pass
    # ensure the input value is in the valid range
    value = max(input_max, min(input_min, value))
    
    # linear mapping conversion
    converted_value = output_min + (output_max - output_min) * (input_min - value) / (input_min - input_max)
    
    converted_value = round(converted_value, 3)
    return converted_value