from unstructured.partition.auto import partition
from unstructured.documents.elements import NarrativeText
from unstructured.partition.text_type import sentence_count
from unstructured.staging.base import elements_to_json, elements_from_json

input_filename = "data/2014 A Review of Dynamic Material Flow Analysis Methods.pdf"
output_filename = "data/output/outputs.json"

elements = partition(filename=input_filename, strategy="hi_res")

elements_to_json(elements, filename=output_filename)


# elements = elements_from_json(filename=filename)

# for element in elements[:100]:
#     if isinstance(element, NarrativeText) and sentence_count(element.text) > 2:
#         print(element)
#         print("\n")
