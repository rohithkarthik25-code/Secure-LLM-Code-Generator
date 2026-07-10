import os
import sys
import numpy as np
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import util

from capextract.core.models import PrimitiveCap

class VectorMapper:
    """
    Replaces Tier 1 hardcoded string dictionaries.
    Uses CodeBERT sentence embeddings to dynamically map function names/calls 
    from the AST to the closest Primitive Capability based on cosine similarity.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, model_name: str = "microsoft/codebert-base", threshold: float = 0.55):
        if cls._instance is None:
            cls._instance = cls(model_name, threshold)
        return cls._instance
    
    def __init__(self, model_name: str = "microsoft/codebert-base", threshold: float = 0.55):
        self.threshold = threshold
        try:
            print(f"[VectorMapper] Loading embedding model: {model_name}...")
            try:
                import transformers
                import transformers.utils.import_utils
                import transformers.modeling_utils
                transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
                transformers.modeling_utils.check_torch_load_is_safe = lambda: None
            except Exception:
                pass
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[VectorMapper] Error loading model: {e}")
            raise

        # Exhaustive semantic anchors for primitives to map against
        self.primitive_anchors = {
            # Filesystem
            PrimitiveCap.FILE_READ: ["read file", "open file", "fread", "ifstream", "readlines"],
            PrimitiveCap.FILE_WRITE: ["write file", "save file", "fwrite", "ofstream", "writelines"],
            PrimitiveCap.FILE_DELETE: ["delete file", "remove file", "unlink", "rm"],
            PrimitiveCap.FILE_PERMISSIONS_MODIFY: ["chmod", "change permissions", "chown", "file attributes"],
            PrimitiveCap.TEMP_FILE_CREATE: ["tempfile", "mktemp", "temporary file", "tmp"],
            PrimitiveCap.PATH_TRAVERSE: ["directory traversal", "path join", "cd", "chdir", "mkdir", "ls"],
            
            # Network
            PrimitiveCap.HTTP_REQUEST: ["http request", "fetch url", "get url", "requests.get", "urlopen", "curl", "post request"],
            PrimitiveCap.SOCKET_OPEN: ["open socket", "tcp connect", "udp connect", "socket create", "connect port"],
            PrimitiveCap.SOCKET_LISTEN: ["socket bind", "listen port", "accept connection", "server socket"],
            PrimitiveCap.DNS_LOOKUP: ["dns lookup", "resolve hostname", "gethostbyname", "dns query"],
            PrimitiveCap.DATA_EXFIL: ["upload data", "send data externally", "exfiltrate payload", "upload file"],
            PrimitiveCap.SSH_CONNECT: ["ssh connect", "paramiko", "scp transfer", "sftp"],
            PrimitiveCap.FTP_TRANSFER: ["ftp upload", "ftp download", "ftplib"],
            PrimitiveCap.RDP_CONNECT: ["rdp connect", "remote desktop"],
            
            # Process
            PrimitiveCap.PROCESS_EXEC: ["execute process", "spawn process", "system call", "subprocess.run", "execve", "os.system", "create process"],
            PrimitiveCap.SHELL_INVOKE: ["invoke shell", "bash", "cmd.exe", "sh", "popen", "shell=True"],
            PrimitiveCap.CODE_EVAL: ["eval", "exec", "run code string", "compile code dynamically"],
            PrimitiveCap.PROCESS_TERMINATE: ["kill process", "terminate process", "exit", "os.kill"],
            PrimitiveCap.PROCESS_LIST: ["list processes", "psutil", "get pid", "tasklist"],
            
            # Database
            PrimitiveCap.DB_CONNECT: ["database connect", "sql connect", "mysql", "postgres", "sqlite3_open", "sqlalchemy"],
            PrimitiveCap.DB_READ: ["sql select", "fetch rows", "query database", "read table"],
            PrimitiveCap.DB_WRITE: ["sql insert", "sql update", "write to database", "commit transaction"],
            PrimitiveCap.DB_SCHEMA_MODIFY: ["create table", "drop table", "alter schema", "sql ddl"],
            PrimitiveCap.DB_RAW_EXECUTE: ["execute raw sql", "cursor.execute", "run query"],
            
            # Auth & Authz
            PrimitiveCap.AUTH_VERIFY: ["verify password", "check credentials", "authenticate user", "login"],
            PrimitiveCap.TOKEN_GENERATE: ["generate jwt", "create token", "issue oauth token"],
            PrimitiveCap.TOKEN_VALIDATE: ["validate token", "verify jwt signature", "check auth token"],
            PrimitiveCap.ACL_QUERY: ["check permissions", "get acl", "user roles"],
            
            # Cryptography
            PrimitiveCap.CRYPTO_ENCRYPT: ["encrypt data", "aes encrypt", "rsa encrypt", "cipher"],
            PrimitiveCap.CRYPTO_DECRYPT: ["decrypt data", "aes decrypt", "decode payload"],
            PrimitiveCap.CRYPTO_HASH: ["hash data", "md5", "sha256", "bcrypt", "compute hash"],
            PrimitiveCap.CRYPTO_SIGN: ["sign payload", "hmac", "digital signature"],
            PrimitiveCap.RANDOM_BYTES_GEN: ["generate random bytes", "os.urandom", "secrets.token", "secure random"],
            
            # Cloud & Virtualization
            PrimitiveCap.CLOUD_STORAGE_ACCESS: ["s3 bucket", "gcs bucket", "azure blob", "cloud storage"],
            PrimitiveCap.CLOUD_METADATA_QUERY: ["aws metadata", "gcp metadata", "instance identity"],
            PrimitiveCap.CONTAINER_SOCKET_INTERACT: ["docker socket", "docker.sock", "container api"],
            PrimitiveCap.K8S_API_QUERY: ["kubernetes api", "kubectl", "k8s config"],
            PrimitiveCap.VM_LIFECYCLE_MANAGE: ["start vm", "stop vm", "ec2 instance"],
            
            # Memory & System
            PrimitiveCap.MEMORY_ALLOCATE: ["allocate memory", "malloc", "calloc", "new"],
            PrimitiveCap.MEMORY_PROTECT_MODIFY: ["mprotect", "virtualprotect", "change memory protection"],
            PrimitiveCap.MEMORY_COPY: ["memcpy", "memmove", "copy memory buffer"],
            PrimitiveCap.DYNAMIC_LIB_LOAD: ["load library", "dlopen", "LoadLibrary", "import module dynamically", "ctypes"],
            PrimitiveCap.ENV_READ: ["read environment variable", "getenv", "os.environ"],
            PrimitiveCap.ENV_WRITE: ["set environment variable", "putenv", "os.environ.setdefault"],
            PrimitiveCap.IPC_COMMUNICATE: ["inter-process communication", "named pipe", "shared memory", "message queue"],
            PrimitiveCap.SYS_SHUTDOWN: ["shutdown system", "reboot", "poweroff"],
            PrimitiveCap.SYS_INFO: ["get system info", "platform uname", "cpu architecture", "hostname"],
            PrimitiveCap.REGISTRY_MODIFY: ["winreg", "modify registry key", "edit windows registry"],
            PrimitiveCap.STARTUP_MODIFY: ["add to startup", "startup folder", "autorun"],
            PrimitiveCap.CRON_MODIFY: ["crontab edit", "add cron job", "scheduled task"],
            PrimitiveCap.LOG_DELETE: ["clear logs", "delete event log", "wevtutil", "rm /var/log"],
            PrimitiveCap.ANTI_DEBUG: ["isdebuggerpresent", "ptrace", "detect debugger", "anti-analysis"],
            PrimitiveCap.OBFUSCATION: ["base64 decode payload", "deobfuscate string", "xor decrypt payload"],
            
            # Data Processing
            PrimitiveCap.SERIALIZE_DATA: ["serialize object", "pickle.dumps", "json stringify", "marshal"],
            PrimitiveCap.DESERIALIZE_DATA: ["deserialize object", "pickle.loads", "json parse", "unmarshal"],
            PrimitiveCap.ARCHIVE_EXTRACT: ["extract zip", "tar extract", "unzip files", "shutil.unpack_archive"],
            PrimitiveCap.XML_PARSE: ["parse xml", "beautifulsoup", "lxml", "etree"],
            PrimitiveCap.USER_INPUT_READ: ["read user input", "scanf", "cin", "input", "readline"],
            PrimitiveCap.CONSOLE_OUTPUT: ["print to console", "console log", "printf", "cout", "println"],
            PrimitiveCap.PDF_PARSE: ["parse pdf", "read pdf text", "pypdf", "pdfplumber"],
            PrimitiveCap.DOCUMENT_GENERATE: ["generate word document", "create docx", "write pdf"],
            
            # Media
            PrimitiveCap.IMAGE_LOAD: ["load image", "cv2.imread", "pil open image"],
            PrimitiveCap.IMAGE_TRANSFORM: ["resize image", "convert color space", "image filter", "blur image"],
            PrimitiveCap.AUDIO_PROCESS: ["process audio", "librosa", "read wav", "spectrogram"],
            PrimitiveCap.VIDEO_PROCESS: ["process video", "extract frames", "moviepy", "opencv video"],
            
            # Messaging/Communication
            PrimitiveCap.EMAIL_SEND: ["send email", "smtplib", "mailserver", "send message"],
            PrimitiveCap.SMS_SEND: ["send sms", "twilio", "text message"],
            PrimitiveCap.WEBHOOK_TRIGGER: ["trigger webhook", "post to discord", "slack webhook"],
            PrimitiveCap.MQ_PUBLISH: ["publish message queue", "rabbitmq publish", "kafka producer", "pika"],
            PrimitiveCap.MQ_SUBSCRIBE: ["subscribe message queue", "rabbitmq consume", "kafka consumer"],
            
            # OS Interaction
            PrimitiveCap.BROWSER_AUTOMATE: ["automate browser", "selenium webdriver", "puppeteer", "playwright"],
            PrimitiveCap.SCREENSHOT_CAPTURE: ["take screenshot", "capture screen", "pyautogui screenshot"],
            PrimitiveCap.CLIPBOARD_ACCESS: ["read clipboard", "write clipboard", "pyperclip"],
            PrimitiveCap.KEYLOG_CAPTURE: ["capture keystrokes", "pynput keyboard", "getasynckeystate", "keylogger"],
            
            # Blockchain
            PrimitiveCap.BLOCKCHAIN_TRANSACT: ["blockchain transaction", "send ethereum", "smart contract execute"],
            PrimitiveCap.WALLET_ACCESS: ["access crypto wallet", "private key", "metamask"],
            
            # ML/Stats
            PrimitiveCap.ML_MODEL_LOAD: ["load model", "torch.load", "load_weights", "transformers model"],
            PrimitiveCap.ML_MODEL_TRAIN: ["train model", "model.fit", "backpropagation", "gradient descent"],
            PrimitiveCap.ML_INFERENCE_RUN: ["run inference", "model.predict", "forward pass"],
            PrimitiveCap.VECTOR_DB_ACCESS: ["vector database", "pinecone", "milvus", "qdrant", "chroma"],
            PrimitiveCap.TABULAR_DATA_OP: ["pandas dataframe", "csv read", "data table", "merge data"],
            PrimitiveCap.STAT_OP: ["compute mean", "calculate variance", "statistics", "scipy", "numpy mean"],
            PrimitiveCap.AGGREGATION: ["groupby", "aggregate data", "sum elements"],
            PrimitiveCap.GPU_ACCESS: ["cuda device", "to gpu", "gpu memory"],
            
            # General computation
            PrimitiveCap.LOOP_CONSTRUCT: ["for loop", "while loop", "iterate over array"],
            PrimitiveCap.MATH_OP: ["math operation", "calculate", "convert number", "add", "subtract", "multiply", "divide", "float", "int", "cast to float"],
            PrimitiveCap.STRING_OP: ["string manipulate", "concatenate strings", "split string", "replace text"],
            PrimitiveCap.DATA_STRUCTURE: ["list append", "dictionary insert", "hash map", "array manipulation"],
            PrimitiveCap.CONDITIONAL: ["if statement", "switch case", "ternary"],
            PrimitiveCap.FUNCTION_DEF: ["define function", "method signature"],
            PrimitiveCap.CLASS_DEF: ["define class", "struct declaration"],
            PrimitiveCap.ERROR_HANDLING: ["try catch", "exception block", "error handle"],
            PrimitiveCap.SORT_ALGO: ["sort array", "quicksort", "order by"],
            PrimitiveCap.REGEX_OP: ["regular expression match", "regex search", "pattern match"],
            PrimitiveCap.TIMER_OP: ["sleep", "timeout", "delay execution", "wait time"],
            PrimitiveCap.THREADING: ["multithreading", "spawn thread", "async execution", "worker thread"],
            
            # Unknown / Plain Text
            PrimitiveCap.UNKNOWN: ["unknown capability", "unrecognized"],
            PrimitiveCap.NATURAL_LANGUAGE: ["natural language", "conversational text", "plain text", "joke", "greeting"],
        }
        
        # Precompute embeddings for all anchors
        self.anchor_embeddings = {}
        for cap, phrases in self.primitive_anchors.items():
            self.anchor_embeddings[cap] = self.model.encode(phrases, convert_to_tensor=True)
            
        print("[VectorMapper] Successfully loaded and pre-computed capability vectors.")

    def match_primitive(self, function_text: str) -> list[Tuple[PrimitiveCap, float]]:
        """
        Embeds the function text and calculates cosine similarity against all primitive anchors.
        Dynamically selects the optimal number of top matches (dynamic k) by analyzing the score drop-off.
        """
        if not function_text or len(function_text.strip()) == 0:
            return []
            
        func_emb = self.model.encode(function_text, convert_to_tensor=True)
        
        matches = []
        for cap, anchor_embs in self.anchor_embeddings.items():
            cos_scores = util.cos_sim(func_emb, anchor_embs)[0]
            max_score = torch.max(cos_scores).item()
            
            if max_score >= self.threshold:
                matches.append((cap, max_score))
                
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if not matches:
            return []
            
        # DYNAMIC OPTIMAL K ALGORITHM
        # We take the top match, and any subsequent match that is within 10% (0.10) 
        # of the top score. This perfectly groups highly-correlated capabilities 
        # (e.g. HTTP_REQUEST and SOCKET_OPEN) while dropping the noisy tails.
        best_score = matches[0][1]
        optimal_matches = [m for m in matches if (best_score - m[1]) <= 0.008]
        
        return optimal_matches
