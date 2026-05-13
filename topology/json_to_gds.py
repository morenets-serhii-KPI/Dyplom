import json
import gdstk


def convert_json_to_gds(
    input_json,
    output_gds
):

    lib = gdstk.Library()

    cell = lib.new_cell("TOP")

    with open(input_json) as f:

        data = json.load(f)

    for poly in data:

        polygon = gdstk.Polygon(

            poly["points"],

            layer=poly["layer"],

            datatype=poly["datatype"]
        )

        cell.add(polygon)

    lib.write_gds(output_gds)

    print(
        f"Saved: {output_gds}"
    )


convert_json_to_gds(
    "sweep_input.json",
    "sweep_input.gds"
)

convert_json_to_gds(
    "sweep_result.json",
    "sweep_result.gds"
)