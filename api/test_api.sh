#!/bin/bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d @/home/zzh/webpage-classification/test_request.json
