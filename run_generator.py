from ascii_generator import ASCII_Generator

# Make sure the file you intend to convert is in the same directory as run_generator.py & ascii_generator.py

""" Step 1: Initializing the generator object
    for example:

    gen = ASCII_Generator()
"""

"""
    Step 2: Calling class-methods to run the conversion
    convert_img works with .jpg & .png
    the output image can be .png(default)
    for example: 

    gen.convert_img("example.jpg", output_file="example_out.png", scale=0.08)

    convert_vid works with .mp4 & .mov
    the output video can be .mp4(default)
    for example: gen.convert_vid("example.mov")

    gen.convert_vid("example.mov", output_file="example_out.mp4", scale=0.05)
"""

gen = ASCII_Generator()
gen.convert_vid("shimmer.mp4", "shimmer_out.mp4")
# gen.convert_vid("spinning_dodecahedron.mp4", "spinning_dodecahedron_out.mp4")
# gen.convert_img("Pilot Wallpaper.png", "Pilot Wallpaper_out.png")
