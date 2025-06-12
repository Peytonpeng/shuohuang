from flask import Blueprint
from auth import token_required, jwt_token, check_system_token

hello_blueprint = Blueprint('hello', __name__)


@hello_blueprint.route('/api/hello')
@token_required
def hello():
    return "hello World"