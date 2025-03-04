from src.line import *

if __name__ == "__main__":
    # Build the line
    line_builder = OverheadLineBuilder()
    line_builder.add_conductor(0.3060, 0.0244, 0.0, 29.0, 0.721)  # Phase A
    line_builder.add_conductor(0.3060, 0.0244, 2.5, 29.0, 0.721)  # Phase B
    line_builder.add_conductor(0.3060, 0.0244, 7.0, 29.0, 0.721)  # Phase C
    line_builder.add_conductor(0.5920, 0.0081, 4.0, 25.0, 0.563)  # Neutral
    line_builder.build()

    #Raw values from constructed line
    print("Z matrix:")
    print(line_builder.Z)
    print("Y matrix:")
    print(line_builder.Y)
    print("tn matrix:")
    print(line_builder.tn)

    # Create the line object
    line1 = LineObject(line_builder)

    #Print A, B, a,b matrices
    print("A matrix:")
    print(line1.A)
    print("B matrix:")
    print(line1.B)
    print("a matrix:")
    print(line1.a)
    print("b matrix:")
    print(line1.b)
    '''
    cable_builder = UndergroundConcentricCableBuilder()
    cable_builder.add_phase_conductor(0.0511, 0.0081, 0.567)  # Phase A
    cable_builder.add_concentric_neutral(0.0100, 0.0081, 0.0641, 0.2835, 13)  # Neutral with 13 strands
    cable_builder.build()

    cable1 = UndergroundConcentricCableObject(cable_builder)
    '''

 
