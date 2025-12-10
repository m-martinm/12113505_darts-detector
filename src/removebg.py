import rembg
from pathlib import Path

if __name__ == "__main__":
    script_location = Path(__file__).parent
    input_dir = (script_location / "../../data/sift").resolve()
    output_dir = input_dir / "bg_removed"
    output_dir.mkdir(exist_ok=True)

    for f in input_dir.glob("*.jpg"):
        with open(f, 'rb') as i:
            with open(output_dir / f.name, 'wb') as o:
                input = i.read()
                output = rembg.remove(input)
                o.write(output)