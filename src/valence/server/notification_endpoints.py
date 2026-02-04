"""Notification API endpoints.

Implements:
- GET /api/v1/notifications - List pending notifications for a recipient
- POST /api/v1/notifications/{id}/acknowledge - Acknowledge a notification
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from .sharing_endpoints import get_sharing_service

logger = logging.getLogger(__name__)


async def list_notifications_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/notifications - List pending notifications for a recipient.
    
    Query Parameters:
        recipient_did: The DID of the recipient (required)
        
    Returns:
        200: List of pending notifications
        400: Missing recipient_did parameter
        503: Service not initialized
    """
    service = get_sharing_service()
    if service is None:
        return JSONResponse(
            {"error": "Sharing service not initialized"},
            status_code=503,
        )
    
    recipient_did = request.query_params.get("recipient_did")
    if not recipient_did:
        return JSONResponse(
            {"error": "recipient_did query parameter is required"},
            status_code=400,
        )
    
    try:
        notifications = await service.get_pending_notifications(recipient_did)
        
        return JSONResponse(
            {
                "success": True,
                "notifications": [_notification_to_dict(n) for n in notifications],
                "count": len(notifications),
            },
            status_code=200,
        )
        
    except Exception as e:
        logger.exception(f"Error listing notifications: {e}")
        return JSONResponse(
            {"error": "Internal server error", "success": False},
            status_code=500,
        )


async def acknowledge_notification_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/notifications/{id}/acknowledge - Acknowledge a notification.
    
    Path Parameters:
        id: Notification UUID
        
    Request Body (JSON):
        {
            "recipient_did": "did:key:..."  // Required for authorization
        }
        
    Returns:
        200: Notification acknowledged successfully
        400: Invalid request (missing recipient_did)
        403: Not the notification recipient
        404: Notification not found
        500: Server error
    """
    service = get_sharing_service()
    if service is None:
        return JSONResponse(
            {"error": "Sharing service not initialized"},
            status_code=503,
        )
    
    notification_id = request.path_params.get("id")
    if not notification_id:
        return JSONResponse(
            {"error": "Notification ID is required"},
            status_code=400,
        )
    
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"},
            status_code=400,
        )
    
    recipient_did = body.get("recipient_did")
    if not recipient_did:
        return JSONResponse(
            {"error": "recipient_did is required"},
            status_code=400,
        )
    
    try:
        success = await service.acknowledge_notification(notification_id, recipient_did)
        
        if not success:
            return JSONResponse(
                {"error": "Notification not found", "success": False},
                status_code=404,
            )
        
        return JSONResponse(
            {
                "success": True,
                "notification_id": notification_id,
                "acknowledged": True,
            },
            status_code=200,
        )
        
    except PermissionError as e:
        return JSONResponse(
            {"error": str(e), "success": False},
            status_code=403,
        )
    except Exception as e:
        logger.exception(f"Error acknowledging notification: {e}")
        return JSONResponse(
            {"error": "Internal server error", "success": False},
            status_code=500,
        )


def _notification_to_dict(notification: Any) -> dict[str, Any]:
    """Convert a Notification to a JSON-serializable dict."""
    return {
        "id": notification.id,
        "recipient_did": notification.recipient_did,
        "notification_type": notification.notification_type,
        "payload": notification.payload,
        "created_at": notification.created_at,
        "acknowledged_at": notification.acknowledged_at,
    }
