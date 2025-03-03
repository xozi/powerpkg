from src.line import *

if __name__ == "__main__":
    # Build the line
    line_builder = OverheadLineBuilder()
    line_builder.add_conductor(0.3060, 0.0244, 0.0, 29.0, 0.721)  # Phase A
    line_builder.add_conductor(0.3060, 0.0244, 2.5, 29.0, 0.721)  # Phase B
    line_builder.add_conductor(0.3060, 0.0244, 7.0, 29.0, 0.721)  # Phase C
    line_builder.add_conductor(0.5920, 0.0081, 4.0, 25.0, 0.563)  # Neutral
    line_builder.build()

    # Create the line object
    line1 = OverheadLineObject(line_builder)

    # Build the underground cable
    cable_builder = UndergroundConcentricCableBuilder()
    cable_builder.add_phase_conductor(0.0511, 0.0081, 0.567)  # Phase A
    cable_builder.add_concentric_neutral(0.0100, 0.0081, 0.0641, 0.2835, 13)  # Neutral with 13 strands
    cable_builder.build()

    # Create the cable object
    cable1 = UndergroundConcentricCableObject(cable_builder)

    # Print the matrices
    print("Z matrix:")
    print(line1.Z)
    print("Y matrix:")
    print(line1.Y)
    print("tn matrix:")
    print(line1.tn)

    print("Z matrix:")
    print(cable1.Z)
    print("Y matrix:")
    print(cable1.Y)
    print("tn matrix:")
    print(cable1.tn)