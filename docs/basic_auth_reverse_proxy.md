# Basic Auth via Reverse Proxy

This guide shows how to protect the LexiAI UI and API with Basic Auth at a
reverse proxy. Use this when you want a simple perimeter gate in front of
the FastAPI service.

Notes
- This is an alternative to app-level auth. You can also layer both.
- The examples assume LexiAI listens on 127.0.0.1:8000.

## Nginx

1) Create an htpasswd file

```bash
htpasswd -c /etc/nginx/lexi.htpasswd lexi
```

2) Configure a server block

```nginx
server {
    listen 80;
    server_name lexiai.local;

    location / {
        auth_basic "LexiAI";
        auth_basic_user_file /etc/nginx/lexi.htpasswd;

        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Caddy

1) Create a hashed password

```bash
caddy hash-password --plaintext "your_password"
```

2) Configure Caddyfile

```caddy
lexiai.local {
    basicauth /* {
        lexi JDJhJDE0JG1qS3FpS0tXb2l4UlY2S09yQjM1N2V5b0E5YzBvMmY2c3h4R2dPNHh4Y3QxSm5XcXhNRGhp
    }

    reverse_proxy 127.0.0.1:8000
}
```

## Optional: Layer with UI auth

If you want to keep UI auth enabled in addition to Basic Auth, set:

```
LEXI_UI_AUTH_REQUIRED=true
```

