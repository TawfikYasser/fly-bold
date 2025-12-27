#!/bin/bash

# Generate self-signed TLS certificates for Flower
# This script creates CA, server, and client certificates

set -e

CERT_DIR="./certs"
CA_KEY="$CERT_DIR/ca.key"
CA_CERT="$CERT_DIR/ca.crt"
SERVER_KEY="$CERT_DIR/server.key"
SERVER_CERT="$CERT_DIR/server.crt"
CLIENT_KEY="$CERT_DIR/client.key"
CLIENT_CERT="$CERT_DIR/client.crt"

# Create certs directory if it doesn't exist
mkdir -p "$CERT_DIR"

echo "Generating CA private key..."
openssl genrsa -out "$CA_KEY" 2048

echo "Generating CA certificate..."
openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 -out "$CA_CERT" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=FlowerCA"

echo "Generating server private key..."
openssl genrsa -out "$SERVER_KEY" 2048

echo "Generating server certificate signing request..."
openssl req -new -key "$SERVER_KEY" -out "$CERT_DIR/server.csr" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

echo "Signing server certificate..."
openssl x509 -req -in "$CERT_DIR/server.csr" -CA "$CA_CERT" -CAkey "$CA_KEY" \
    -CAcreateserial -out "$SERVER_CERT" -days 3650 -sha256

echo "Generating client private key..."
openssl genrsa -out "$CLIENT_KEY" 2048

echo "Generating client certificate signing request..."
openssl req -new -key "$CLIENT_KEY" -out "$CERT_DIR/client.csr" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=client"

echo "Signing client certificate..."
openssl x509 -req -in "$CERT_DIR/client.csr" -CA "$CA_CERT" -CAkey "$CA_KEY" \
    -CAcreateserial -out "$CLIENT_CERT" -days 3650 -sha256

# Clean up CSR files
rm -f "$CERT_DIR/server.csr" "$CERT_DIR/client.csr"

echo "Certificates generated successfully in $CERT_DIR"
echo "CA Certificate: $CA_CERT"
echo "Server Certificate: $SERVER_CERT"
echo "Client Certificate: $CLIENT_CERT"