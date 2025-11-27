def validate_param_values(data):
    """
    校验参数结构是否为 [{...}, {...}]
    并检查每个字典是否包含所需键，并限制 param_values 长度不超过 4。
    """
    error_messages = []

    if not isinstance(data, list):
        return False, ["参数格式错误：最外层必须是一个列表"]

    for param_obj in data:
        if not isinstance(param_obj, dict):
            error_messages.append(f"参数格式错误：参数项必须是字典，实际为 {type(param_obj).__name__}")
            continue

        required_keys = ["param_name", "default_value", "param_values"]
        missing_keys = [key for key in required_keys if key not in param_obj]
        if missing_keys:
            error_messages.append(f"参数对象 {param_obj} 缺少键: {', '.join(missing_keys)}")
            continue  # 如果缺键就不再继续校验这个 param_obj

        param_values = param_obj.get('param_values')
        if not isinstance(param_values, list):
            error_messages.append(f"参数 '{param_obj.get('param_name')}' 的 'param_values' 必须是一个列表")
        elif len(param_values) > 4:
            error_messages.append(f"参数 '{param_obj.get('param_name')}' 的 'param_values' 长度不能超过 4，当前为 {len(param_values)}")

    if error_messages:
        return False, error_messages

    return True, []