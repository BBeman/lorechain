from .loremaster import ask_loremaster
from .council import create_lore


running = True

while running:
    choice = input("=== WELCOME TO LORECHAIN ===\nSelect an option:\n#1 Ask the LoreMaster\n#2 Create new Lore\n")
    if choice == "1":
        print(ask_loremaster(input("Your question: ")))
        running = False
    elif choice == "2":
        print(create_lore(input("Enter new Lore to Generate: ")))
        running = False
    else:
        running = True
        print("Invalid option\n")