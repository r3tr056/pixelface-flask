
from werkzeug.security import generate_password_hash, check_password_hash
from src.pixelface.api.extensions import db

class Project(db.Model):
    __tablename__ = 'projects'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    video_hash = db.Column(db.String(100), unique=True, nullable=False)
    video_path = db.Column(db.String(200), nullable=True)
    streaming_video_key = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(50), default='pending')
    models_generated = db.Column(db.Boolean, default=False)

    # Relationships
    faces = db.relationship('Face', backref='project', lazy=True)

    def __repr__(self):
        return f"<Project {self.id}, Status: {self.status}>"

class Face(db.Model):
    __tablename__ = 'faces'
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    thumbnail_path = db.Column(db.String(200), nullable=True)
    texture_path = db.Column(db.String(200), nullable=True)
    obj_path = db.Column(db.String(200), nullable=False)
    mtl_path = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f"<Face {self.id} for Project {self.project_id}>"

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)  # Nullable to support Google login users
    google_user = db.Column(db.Boolean, default=False)

    # Relationships
    projects = db.relationship('Project', backref='user', lazy=True)

    def set_password(self, password):
        """Hashes password for the user."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks hashed password."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.email}>"

class TokenBlacklist(db.Model):
    __tablename__ = 'token_blacklist'
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, index=True, unique=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

    def __repr__(self):
        return f"<TokenBlacklist {self.jti}>"