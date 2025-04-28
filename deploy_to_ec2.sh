#!/bin/bash
# Script to deploy bug detection system to EC2 G5 instance

# Configuration
EC2_USERNAME="ec2-user"
EC2_HOST="$1"  # Pass the EC2 instance IP as the first argument
KEY_PATH="$2"  # Pass the path to your .pem key file as the second argument
REMOTE_DIR="/home/ec2-user/bug_detection"

# Check if the required arguments are provided
if [ -z "$EC2_HOST" ] || [ -z "$KEY_PATH" ]; then
    echo "Usage: ./deploy_to_ec2.sh <ec2-instance-ip> <path-to-pem-key>"
    exit 1
fi

# Create a tar archive of the project (excluding unnecessary files)
echo "Creating project archive..."
tar --exclude='*.tar.gz' --exclude='venv' --exclude='__pycache__' --exclude='*.pyc' -czf bug_detection.tar.gz .

# Ensure the remote directory exists
echo "Setting up remote directory..."
ssh -i "$KEY_PATH" "$EC2_USERNAME@$EC2_HOST" "mkdir -p $REMOTE_DIR"

# Copy the project archive to the EC2 instance
echo "Copying project to EC2 instance..."
scp -i "$KEY_PATH" bug_detection.tar.gz "$EC2_USERNAME@$EC2_HOST:$REMOTE_DIR/"

# Extract the archive on the EC2 instance and set up the environment
echo "Setting up the environment on EC2..."
ssh -i "$KEY_PATH" "$EC2_USERNAME@$EC2_HOST" "cd $REMOTE_DIR && tar -xzf bug_detection.tar.gz && chmod +x ec2_setup.sh && ./ec2_setup.sh"

# Clean up the local archive
echo "Cleaning up..."
rm bug_detection.tar.gz

echo "Deployment completed!"
echo "Connect to your EC2 instance: ssh -i $KEY_PATH $EC2_USERNAME@$EC2_HOST"
echo "Then run: cd $REMOTE_DIR && source venv/bin/activate && python main.py"
