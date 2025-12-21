#!/usr/bin/env python3
"""
Password Reset Script für LexiAI Users
Setzt das Passwort für einen bestimmten User zurück
"""
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.password_utils import hash_password
from backend.utils.user_persistence import UserPersistence


def reset_password(email: str, new_password: str):
    """
    Setzt das Passwort für einen User zurück

    Args:
        email: Email des Users
        new_password: Neues Passwort (muss Policy erfüllen)
    """
    # User laden
    users = UserPersistence.load_users()

    # User finden
    user_id = None
    for uid, user in users.items():
        if user["email"].lower() == email.lower():
            user_id = uid
            break

    if not user_id:
        print(f"❌ Fehler: User mit Email '{email}' nicht gefunden")
        print(f"\nVerfügbare User:")
        for uid, user in users.items():
            print(f"  - {user['email']}")
        return False

    # Passwort hashen
    password_hash = hash_password(new_password)

    # Passwort aktualisieren
    users[user_id]["password_hash"] = password_hash

    # Speichern
    if UserPersistence.save_users(users):
        print(f"✅ Passwort erfolgreich zurückgesetzt für User: {email}")
        print(f"   User-ID: {user_id}")
        print(f"   Neues Passwort: {new_password}")
        print(f"\n⚠️  WICHTIG: Passwort nach Login ändern!")
        return True
    else:
        print(f"❌ Fehler beim Speichern der User-Datenbank")
        return False


if __name__ == "__main__":
    # Default: thomas.sigmund1989@gmail.com mit TestPass123
    email = "thomas.sigmund1989@gmail.com"
    new_password = "TestPass123"  # Erfüllt Policy: 8+ Zeichen, 1 Großbuchstabe, 1 Zahl

    # Optionale CLI-Argumente
    if len(sys.argv) > 1:
        email = sys.argv[1]
    if len(sys.argv) > 2:
        new_password = sys.argv[2]

    print(f"🔐 Passwort-Reset für LexiAI")
    print(f"=" * 50)
    print(f"Email: {email}")
    print(f"Neues Passwort: {new_password}")
    print(f"=" * 50)
    print()

    reset_password(email, new_password)
