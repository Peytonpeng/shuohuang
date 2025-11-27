import datetime
from functools import wraps
import jwt
from flask import request, jsonify
from config import SECRET_KEY

# 权限校验 - 接口wraps装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # 从请求头中获取 token，例如 'Authorization': 'Bearer <token>'
        if 'X-Api-Key' in request.headers:
            token = request.headers['X-Api-Key']
        if not token:
            return jsonify({'error': 'X-Api-Key is missing!'}), 403

        try:
            jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except:
            return jsonify({'error': 'X-Api-Key is invalid!'}), 403

        return f(*args, **kwargs)

    return decorated

# 权限校验 - 根据SystemToken(系统Token)生成subToken(子token)
def jwt_token(SystemToken):
    payload = {
        'user_id': SystemToken,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)  # token 过期时间
    }
    # 生成 token
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return {'token': token,'code':200} #.decode('UTF-8')

# 检查系统token
def check_system_token(SystemToken):
    if SystemToken is None:
        return False
    else:
        return True