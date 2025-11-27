# app/routes/auth.py
from flask import Blueprint, request, jsonify, session

from analysis_service.app.utils.auth import token_required, jwt_token, check_system_token
from analysis_service.app.extensions import logger

auth_bp = Blueprint("auth", __name__)

@auth_bp.route('/login/token', methods=['POST'])
def token():
    # 获得系统token
    SystemToken = request.headers.get('SystemToken')
    # print('SystemToken'+SystemToken)
    flag = check_system_token(SystemToken)
    # 验证系统token
    if flag is False:
        return jsonify({'error': 'Invalid system token'}), 401
    else:
        value = jwt_token(SystemToken)
        # 将token保存到session作为默认的WebSocket room id
        session['room_id'] = value['token']
        logger.info(f"用户登录成功，设置room ID为: {value['token']}")
        return jsonify(value)
