import os
import smtplib
from email.message import EmailMessage
from typing import Optional


class EmailService:
    """Gmail SMTP email service using App Password authentication."""
    
    def __init__(self):
        """Initialize email service with environment variables."""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.system_email = os.getenv("SYSTEM_EMAIL", self.smtp_user)
        
        if not self.smtp_user or not self.smtp_pass:
            raise ValueError(
                "SMTP credentials not configured. "
                "Please set SMTP_USER and SMTP_PASS environment variables."
            )
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        reply_to: Optional[str] = None
    ) -> None:
        """
        Send email via Gmail SMTP.
        
        Args:
            to_email: Recipient email address (department email)
            subject: Email subject
            body: Email body content
            reply_to: Student email for Reply-To header (optional)
            
        Raises:
            ValueError: If required fields are missing
            smtplib.SMTPException: If email sending fails
        """
        if not to_email:
            raise ValueError("Recipient email is required")
        if not subject:
            raise ValueError("Subject is required")
        if not body:
            raise ValueError("Body is required")
        
        # Create email message
        msg = EmailMessage()
        msg["From"] = self.system_email
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Add Reply-To header if provided
        if reply_to:
            msg["Reply-To"] = reply_to
        
        # Set email body
        msg.set_content(body)
        
        # Send email via SMTP
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
                print(f"📧 Email sent successfully to {to_email}")
        except smtplib.SMTPAuthenticationError as e:
            error_msg = (
                f"📧 Gmail Authentication Failed (Error 535):\n"
                f"   Your Gmail credentials were rejected.\n\n"
                f"🔧 To fix this:\n"
                f"   1. Make sure you're using a Gmail App Password (NOT your regular password)\n"
                f"   2. Generate an App Password:\n"
                f"      - Go to: https://myaccount.google.com/apppasswords\n"
                f"      - Enable 2-Step Verification first if not enabled\n"
                f"      - Select 'Mail' as the app type\n"
                f"      - Copy the 16-character password (no spaces)\n"
                f"   3. Update your .env file:\n"
                f"      SMTP_USER=your_email@gmail.com\n"
                f"      SMTP_PASS=xxxx xxxx xxxx xxxx  (16-char App Password)\n"
                f"      SYSTEM_EMAIL=your_email@gmail.com\n"
                f"   4. Restart your backend server after updating .env\n\n"
                f"   Current SMTP_USER: {self.smtp_user}\n"
                f"   Make sure SMTP_PASS is an App Password, not your regular Gmail password!"
            )
            print(error_msg)
            raise ValueError(error_msg) from e
        except smtplib.SMTPException as e:
            print(f"📧 SMTP error: {e}")
            raise
        except Exception as e:
            print(f"📧 Unexpected error sending email: {e}")
            raise

