#!/bin/bash

# Generate self-signed TLS certificates for FEDn
set -euo pipefail

CERT_DIR="./certs"
CA_KEY="$CERT_DIR/ca.key"
CA_CERT="$CERT_DIR/ca.crt"
SERVER_KEY="$CERT_DIR/server.key"
SERVER_CERT="$CERT_DIR/server.crt"
CLIENT_KEY="$CERT_DIR/client.key"
CLIENT_CERT="$CERT_DIR/client.crt"

mkdir -p "$CERT_DIR"

echo "Generating CA key..."
openssl genrsa -out "$CA_KEY" 2048

echo "Generating CA cert..."
openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 -out "$CA_CERT" \
  -subj "/C=US/ST=State/L=City/O=FEDn/CN=FEDn-CA"

echo "Generating server key..."
openssl genrsa -out "$SERVER_KEY" 2048

echo "Generating server CSR..."
openssl req -new -key "$SERVER_KEY" -out "$CERT_DIR/server.csr" \
  -subj "/C=US/ST=State/L=City/O=FEDn/CN=server"

echo "Signing server cert..."
openssl x509 -req -in "$CERT_DIR/server.csr" -CA "$CA_CERT" -CAkey "$CA_KEY" \
  -CAcreateserial -out "$SERVER_CERT" -days 3650 -sha256

echo "Generating client key..."
openssl genrsa -out "$CLIENT_KEY" 2048

echo "Generating client CSR..."
openssl req -new -key "$CLIENT_KEY" -out "$CERT_DIR/client.csr" \
  -subj "/C=US/ST=State/L=City/O=FEDn/CN=client"

echo "Signing client cert..."
openssl x509 -req -in "$CERT_DIR/client.csr" -CA "$CA_CERT" -CAkey "$CA_KEY" \
  -CAcreateserial -out "$CLIENT_CERT" -days 3650 -sha256

rm -f "$CERT_DIR/server.csr" "$CERT_DIR/client.csr"

echo "Certificates generated in $CERT_DIR"
echo "CA: $CA_CERT"
echo "Server: $SERVER_CERT"
echo "Client: $CLIENT_CERT"
