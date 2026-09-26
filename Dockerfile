FROM python:3.10-slim

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the current directory contents into the container
COPY --chown=user . $HOME/app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Run the Flask app on port 7860 using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "the_app:app"]
