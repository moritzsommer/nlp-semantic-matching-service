"""Module to extract the classification classes from ECLASS XML files into CSV format."""

import logging
import xml.etree.ElementTree as et

import pandas as pd

from src.utils.logger import LoggerFactory


def extract_eclass_xml(input_path: str, logger: logging.Logger) -> dict:
    """Parses an ECLASS XML file and extracts classification class IDs, names, and definitions into a dict."""

    # Load ECLASS classes from XML file
    logger.info(f"Processing XML: {input_path}")
    try:
        tree = et.parse(input_path)
        root = tree.getroot()
        logger.info(f"Database loaded from {input_path}.")
    except Exception as e:
        logger.error(f"Failed to parse XML {input_path}: {e}")
        return {}

    # Namespaces
    ns = {
        "dic": "urn:eclass:xml-schema:dictionary:5.0",
        "ontoml": "urn:iso:std:iso:is:13584:-32:ed-1:tech:xml-schema:ontoml",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    data: dict = {}

    def extract_elements(elements):
        """
        Extracts fields "id", "preferred_name" and "definition" from "class" nodes of type
        "ontoml:CATEGORIZATION_CLASS_Type" (classification class) and stores them in a dict.
        """

        for elem in elements:
            # Only keep class nodes of type "ontoml:CATEGORIZATION_CLASS_Type"
            xsi_type = elem.attrib.get(f"{{{ns['xsi']}}}type")
            # change in here for checking properties
            if xsi_type != "ontoml:CATEGORIZATION_CLASS_Type":
                continue

            elem_id = elem.attrib.get("id")
            if elem_id is None:
                continue

            # Extract "preferred_name" and "definition" for each classification class
            pref_el = elem.find("preferred_name/label")
            preferred_name = pref_el.text.strip() if pref_el is not None and pref_el.text else None
            def_el = elem.find("definition/text")
            definition = def_el.text.strip() if def_el is not None and def_el.text else None

            # Save results
            data[elem_id] = {
                "preferred-name": preferred_name,
                "definition": definition
            }

    # Extract data from the dictionary
    dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=ns)
    if dictionary_node is not None:
        for contained in dictionary_node.findall(".//contained_classes", namespaces=ns):
            extract_elements(contained)
    else:
        logger.warning(f"No dictionary node found in {input_path}")

    return data


if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    segments = list(range(13, 52)) + [90]
    all_data = {}

    # Run for each segment
    for segment in segments:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue
        input_path = f"../../data/original/ECLASS15_0_BASIC_EN_SG_{segment}.xml"
        segment_data = extract_eclass_xml(input_path, logger)
        all_data.update(segment_data)

        # Save results
        output_path = f"../../data/extracted/eclass-{segment}.csv"
        df = pd.DataFrame.from_dict(segment_data, orient="index").reset_index()
        df.rename(columns={"index": "id"}, inplace=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV to: {output_path}")

    # Save combined results
    output_path = f"../../data/extracted/eclass-all.csv"
    df = pd.DataFrame.from_dict(all_data, orient="index").reset_index()
    df.rename(columns={"index": "id"}, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to: {output_path}")
