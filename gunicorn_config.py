# Gunicorn Configuration File for Medical AI System
# =============================================================================

import multiprocessing
import os

# Server Socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker Processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process Naming
proc_name = "medical_ai_system"

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in future)
# keyfile = None
# certfile = None

# Preload application for better performance
preload_app = True

# Restart workers after this many requests (helps with memory leaks)
max_requests = 1000
max_requests_jitter = 50

print("🔧 Gunicorn configuration loaded successfully")
print(f"👷 Workers: {workers}")
print(f"🌐 Binding to: {bind}")
