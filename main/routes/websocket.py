import datetime
import uuid
import numpy as np
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request, jsonify, Blueprint, logging, app
import sh_analysis.main.Config.getConnection as get_db_connection
import psycopg2

from sh_analysis.main.Main import logger
from sh_analysis.main.routes.train import training_sessions
from sh_analysis.main.utils import feature_extraction
from sh_analysis.main.utils.visualization import Visualizer
from sh_analysis.main.utils.methods import Preprocessor
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

websocket = Blueprint('websocket', __name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域


@socketio.on('connect', namespace='/training')
def handle_training_connect():
    """处理客户端连接"""
    logger.info(f"客户端连接到训练命名空间: {request.sid}")
    emit('connected', {
        'message': '已连接到训练监控',
        'sid': request.sid,
        'timestamp': datetime.datetime.now().isoformat()
    }, room=request.sid, namespace='/training')


@socketio.on('disconnect', namespace='/training')
def handle_training_disconnect():
    """处理客户端断开连接"""
    logger.info(f"客户端断开训练命名空间连接: {request.sid}")


@socketio.on('join_training_room', namespace='/training')
def on_join_training_room(data):
    """处理客户端加入训练房间"""
    training_room_id = data.get('session_id')
    if training_room_id:
        join_room(training_room_id)
        logger.info(f"客户端 {request.sid} 加入房间: {training_room_id}")

        # 发送房间加入成功消息
        emit('room_joined', {
            'message': f'Successfully joined room: {training_room_id}',
            'session_id': training_room_id,
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid, namespace='/training')

        # 如果训练会话存在，发送当前状态
        if training_room_id in training_sessions:
            session_info = training_sessions[training_room_id]
            emit('session_status', {
                'session_id': training_room_id,
                'status': session_info,
                'timestamp': datetime.datetime.now().isoformat()
            }, room=request.sid, namespace='/training')
    else:
        logger.warning(f"客户端 {request.sid} 尝试加入无效房间ID。")
        emit('room_join_error', {
            'message': 'Invalid session_id provided',
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid, namespace='/training')


@socketio.on('leave_training_room', namespace='/training')
def on_leave_training_room(data):
    """处理客户端离开训练房间"""
    training_room_id = data.get('session_id')
    if training_room_id:
        leave_room(training_room_id)
        logger.info(f"客户端 {request.sid} 离开房间: {training_room_id}")
        emit('room_left', {
            'message': f'Left room: {training_room_id}',
            'session_id': training_room_id,
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid, namespace='/training')


@socketio.on('get_session_status', namespace='/training')
def handle_get_session_status(data):
    """获取训练会话状态"""
    session_id = data.get('session_id')
    if session_id and session_id in training_sessions:
        emit('session_status', {
            'session_id': session_id,
            'status': training_sessions[session_id],
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid, namespace='/training')
    else:
        emit('session_status', {
            'session_id': session_id,
            'error': '会话不存在或无效',
            'timestamp': datetime.datetime.now().isoformat()
        }, room=request.sid, namespace='/training')


@socketio.on('get_all_sessions', namespace='/training')
def handle_get_all_sessions():
    """获取所有训练会话状态"""
    emit('all_sessions', {
        'sessions': training_sessions,
        'count': len(training_sessions),
        'timestamp': datetime.datetime.now().isoformat()
    }, room=request.sid, namespace='/training')

