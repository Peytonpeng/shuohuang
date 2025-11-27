# app/websocket.py
import datetime
from flask import request, session
from flask_socketio import join_room, leave_room

from analysis_service.app.extensions import socketio, logger


# === 通用 WebSocket 事件发送函数 ===

def emit_process_progress(room_id, process_type, data):
    """发送通用处理进度消息"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_progress', payload, namespace='/ns_analysis', room=room_id)
        logger.info(f"发送{process_type}进度消息到房间 {room_id} (ns /ns_analysis): {data.get('message', '')}")
    except Exception as e:
        logger.error(f"发送{process_type}进度消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_completed(room_id, process_type, data):
    """发送通用处理完成消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_completed', payload, namespace='/ns_analysis', room=room_id)
        logger.debug(f"发送{process_type}完成消息到房间 {room_id} (ns /ns_analysis)")
    except Exception as e:
        logger.error(f"发送{process_type}完成消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_error(room_id, process_type, error_msg, details=None):
    """发送通用处理错误消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(error_msg)
        }
        if details:
            payload['details'] = str(details)
        socketio.emit('process_error', payload, namespace='/ns_analysis', room=room_id)
        logger.error(f"发送{process_type}错误消息到房间 {room_id} (ns /ns_analysis): {error_msg}")
    except Exception as e:
        logger.error(f"发送{process_type}错误消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_process_result(room_id, process_type, data):
    """发送通用处理中间结果消息"""
    try:
        payload = {
            'room': room_id,
            'process_type': process_type,
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.debug(f"发送{process_type}中间结果到房间 {room_id} (ns /ns_analysis)")
    except Exception as e:
        logger.error(f"发送{process_type}中间结果消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_epoch_result(room_id, data):
    """发送单个 epoch 结果"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': 'training',
            'sub_type': 'epoch_result',
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.debug(f"发送epoch结果到房间 {room_id} (ns /ns_analysis): epoch {data.get('global_epoch', '')}")
    except Exception as e:
        logger.error(f"发送epoch结果消息失败 (房间 {room_id}): {e}", exc_info=True)


def emit_round_result(room_id, data):
    """发送轮次结果"""
    try:
        payload = {
            'room_id': room_id,
            'process_type': 'training',
            'sub_type': 'round_result',
            'timestamp': datetime.datetime.now().isoformat(),
            **data
        }
        socketio.emit('process_result', payload, namespace='/ns_analysis', room=room_id)
        logger.debug(f"发送轮次结果到房间 {room_id} (ns /ns_analysis): round {data.get('current_round', '')}")
    except Exception as e:
        logger.error(f"发送轮次结果消息失败 (房间 {room_id}): {e}", exc_info=True)


def register_websocket_handlers():
    """注册WebSocket事件处理函数"""

    @socketio.on('connect', namespace='/ns_analysis')
    def handle_ns_analysis_connect():
        """处理客户端连接到 /ns_analysis namespace"""
        logger.info(f"客户端 {request.sid} 连接到 /ns_analysis namespace")

    @socketio.on('disconnect', namespace='/ns_analysis')
    def handle_ns_analysis_disconnect():
        """处理客户端从 /ns_analysis namespace 断开连接"""
        room_to_leave = session.get('room_id')
        if room_to_leave:
            leave_room(room_to_leave)
            logger.debug(f"客户端 {request.sid} 自动离开房间 {room_to_leave} (从 /ns_analysis namespace 断开)")
            session.pop('room_id', None)
        else:
            logger.debug(f"客户端 {request.sid} 从 /ns_analysis namespace 断开 (未在 session 中找到房间信息)")

    @socketio.on('join_room', namespace='/ns_analysis')
    def on_join_ns_analysis_room(data):
        """处理客户端请求加入处理房间"""
        room_to_join = data.get('room_id')
        if not isinstance(room_to_join, str) or not room_to_join:
            logger.warning(f"客户端 {request.sid} 尝试加入无效的房间名: {room_to_join}。数据: {data}")
            socketio.emit('room_join_error', {
                'message': '无效的房间名 (room_id) 或未提供。',
                'requested_room': room_to_join,
                'timestamp': datetime.datetime.now().isoformat()
            }, room=request.sid)
            return

        join_room(room_to_join)
        session['room_id'] = room_to_join
        logger.info(f"客户端 {request.sid} 加入房间: {room_to_join} (ns /ns_analysis)")

    @socketio.on('leave_room', namespace='/ns_analysis')
    def on_leave_ns_analysis_room(data):
        """处理客户端主动离开处理房间的请求"""
        room_to_leave = data.get('room_id')
        if not isinstance(room_to_leave, str) or not room_to_leave:
            logger.warning(f"客户端 {request.sid} 尝试离开无效的房间名: {room_to_leave}。数据: {data}")
            return

        leave_room(room_to_leave)
        logger.debug(f"客户端 {request.sid} 主动离开房间: {room_to_leave} (ns /ns_analysis)")

        if session.get('room_id') == room_to_leave:
            session.pop('room_id', None)

    print("WebSocket handlers registered.")
    logger.debug("WebSocket handlers registered.")
