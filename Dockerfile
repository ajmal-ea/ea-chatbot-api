FROM python:3.12.0

WORKDIR /app

# Copy requirements file
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Python application files
COPY backend/ea_chatbot.py .
COPY backend/ea_chatbot_app.py .
COPY backend/prometheus.yml .
COPY backend/entrypoint.sh .

# Create a directory for GeoIP database if used
#RUN mkdir -p /app/geoip

# Optional: Download GeoLite2 City database if you have a license key
# RUN if [ -n "$MAXMIND_LICENSE_KEY" ]; then \
#     apt-get update && apt-get install -y wget && \
#     wget "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&license_key=$MAXMIND_LICENSE_KEY&suffix=tar.gz" -O GeoLite2-City.tar.gz && \
#     tar -xzvf GeoLite2-City.tar.gz && \
#     mv GeoLite2-City_*/GeoLite2-City.mmdb /app/geoip/ && \
#     rm -rf GeoLite2-City.tar.gz GeoLite2-City_* && \
#     apt-get remove -y wget && apt-get autoremove -y && apt-get clean; \
# fi

RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]