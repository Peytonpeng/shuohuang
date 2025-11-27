# analysis_service/utils/common.py

import os
from flask import jsonify


def api_error(state: int, message: str, http_status: int | None = None):
    """
    统一错误返回格式，避免重复写 JSON 和状态码。
    """
    if http_status is None:
        http_status = state
    return jsonify({"state": state, "message": message}), http_status


def api_success(data=None, message="成功", state=200):
    """
    统一成功返回格式，提高命名一致性
    """
    return jsonify({
        "state": state,
        "message": message,
        "data": data
    }), 200


def safe_remove(path: str | None):
    """
    安全删除文件，不会因为删除失败导致主逻辑崩溃
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
