"""Module to extract the classification classes and properties from ECLASS XML files into CSV format."""

import logging
import xml.etree.ElementTree as et

import pandas as pd

from src.utils.logger import LoggerFactory


def extract_eclass_xml(input_path: str, logger: logging.Logger, mode: str = "classes") -> dict:
    """Parses an ECLASS XML file and extracts classification class or property IDs, names and definitions into a
    dict."""

    # Load ECLASS classes from XML file
    logger.info(f"Processing XML: {input_path} (mode={mode})")

    if mode == "properties":
        expected_type = "ontoml:NON_DEPENDENT_P_DET_Type"
        container_xpath = ".//contained_properties"
    elif mode == "classes":
        expected_type = "ontoml:CATEGORIZATION_CLASS_Type"
        container_xpath = ".//contained_classes"
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
        """Extracts fields "id", "preferred_name" and "definition" from a class or property and stores them in a
        dict."""

        for elem in elements:
            # Only keep class nodes of type "ontoml:CATEGORIZATION_CLASS_Type"
            xsi_type = elem.attrib.get(f"{{{ns['xsi']}}}type")
            if xsi_type != expected_type:
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
            new_record = {
                "preferred-name": preferred_name,
                "definition": definition
            }

            if elem_id not in data:
                data[elem_id] = {
                    "primary": new_record,
                    "duplicates": []
                }
            else:
                data[elem_id]["duplicates"].append(new_record)

    # Extract data from the dictionary
    dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=ns)
    if dictionary_node is not None:
        for contained in dictionary_node.findall(container_xpath, namespaces=ns):
            extract_elements(contained)
    else:
        logger.warning(f"No dictionary node found in {input_path}")

    # Summary over duplicates per ID within this segment
    for elem_id, entry in data.items():
        dup_list = entry["duplicates"]
        if not dup_list:
            continue
        logger.warning(
            f"ID {elem_id}: {len(dup_list)} duplicate record(s) within {input_path}"
        )

    return data


if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments
    mode = "classes"  # Categorisation classes via "classes", properties via "properties"

    # Output folder depends on mode
    if mode == "properties":
        out_dir = "../../data/extracted-properties/1-original-properties"
    elif mode == "classes":
        out_dir = "../../data/extracted-classes/1-original-classes"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(f"eclass-{mode}-extraction-run.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    segments = list(range(13, 52)) + [90]
    all_data = {}

    # Run for each segment
    for segment in segments:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue
        input_path = f"../../data/original/ECLASS15_0_BASIC_EN_SG_{segment}.xml"
        segment_data = extract_eclass_xml(input_path, logger, mode)

        # Merge and track duplicates per ID across segments
        for elem_id, entry in segment_data.items():
            primary = entry["primary"]
            dups = entry["duplicates"]
            if elem_id not in all_data:
                all_data[elem_id] = {
                    "primary": primary,
                    "duplicates": list(dups)
                }
            else:
                all_data[elem_id]["duplicates"].append(primary)
                all_data[elem_id]["duplicates"].extend(dups)

        # Save results, duplicate IDs are ignored
        output_path = f"{out_dir}/eclass-{segment}.csv"
        flat_segment = {elem_id: entry["primary"] for elem_id, entry in segment_data.items()}
        df = pd.DataFrame.from_dict(flat_segment, orient="index").reset_index()
        df.rename(columns={"index": "id"}, inplace=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV to: {output_path}")

    # Summary over duplicates per ID over all segments
    total_dup_occ = 0
    total_dup_ids = 0
    for elem_id, entry in all_data.items():
        dup_list = entry["duplicates"]
        if not dup_list:
            continue

        count = len(dup_list)
        total_dup_occ += count
        total_dup_ids += 1

        count = len(dup_list)
        all_identical = all(rec == entry["primary"] for rec in dup_list)

        if all_identical:
            logger.warning(
                f"IRDI {elem_id}: {count} duplicate record(s) over all segments"
                f", all identical to primary entry (mode={mode})"
            )
        else:
            logger.warning(
                f"IRDI {elem_id}: {count} duplicate record(s) over all segments"
                f", at least one conflicting entry (mode={mode})"
            )
    if total_dup_occ > 0:
        logger.warning(
            f"Total duplicate occurrences over all segments across all IRDIs: {total_dup_occ} (mode={mode})"
        )
        logger.warning(
            f"Number of IRDIs over all segments with at least one duplicate: {total_dup_ids} (mode={mode})"
        )
        logger.warning(
            f"Duplicate IRDIs were ignored in the output files, each entry has a unique IRDI (mode={mode})"
        )

    # Save combined results, duplicate IDs are ignored
    output_path = f"{out_dir}/eclass-0.csv"
    flat_all = {elem_id: entry["primary"] for elem_id, entry in all_data.items()}
    df = pd.DataFrame.from_dict(flat_all, orient="index").reset_index()
    df.rename(columns={"index": "id"}, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to: {output_path}")
