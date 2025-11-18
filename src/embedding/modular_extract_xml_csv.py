"""Module to extract classification classes, item classes, and properties from ECLASS XML files into CSV format."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as et

import pandas as pd

from src.utils.logger import LoggerFactory

# Namespaces used in ECLASS XML
NAMESPACES = {
    "dic": "urn:eclass:xml-schema:dictionary:5.0",
    "ontoml": "urn:iso:std:iso:is:13584:-32:ed-1:tech:xml-schema:ontoml",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}


class EClassExtractor:
    """Extracts different types of elements from ECLASS XML files."""

    def __init__(self, logger: logging.Logger):
        """Initialize the extractor with a logger.

        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger
        self.ns = NAMESPACES

    def parse_xml(self, input_path: str) -> Optional[et.Element]:
        """Parse an ECLASS XML file and return the root element.

        Args:
            input_path: Path to the XML file

        Returns:
            Root element of the XML tree, or None if parsing fails
        """
        self.logger.info(f"Processing XML: {input_path}")
        try:
            tree = et.parse(input_path)
            root = tree.getroot()
            self.logger.info(f"Database loaded from {input_path}.")
            return root
        except Exception as e:
            self.logger.error(f"Failed to parse XML {input_path}: {e}")
            return None

    def _extract_text_field(self, elem: et.Element, xpath: str) -> Optional[str]:
        """Extract and strip text from an element using XPath.

        Args:
            elem: Element to search within
            xpath: XPath expression to find the target element

        Returns:
            Stripped text content, or None if not found
        """
        target = elem.find(xpath)
        if target is not None and target.text:
            return target.text.strip()
        return None

    def extract_categorizations(self, root: et.Element) -> Dict[str, Dict[str, Optional[str]]]:
        """Extract categorization classes from the XML.

        Args:
            root: Root element of the XML tree

        Returns:
            Dictionary mapping class IDs to their attributes (preferred_name, definition)
        """
        data = {}
        dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=self.ns)

        if dictionary_node is None:
            self.logger.warning("No dictionary node found")
            return data

        # Find all class elements with CATEGORIZATION_CLASS_Type
        for elem in dictionary_node.findall(".//*[@xsi:type='ontoml:CATEGORIZATION_CLASS_Type']",
                                            namespaces=self.ns):
            elem_id = elem.attrib.get("id")
            if elem_id is None:
                continue

            preferred_name = self._extract_text_field(elem, "preferred_name/label")
            definition = self._extract_text_field(elem, "definition/text")

            data[elem_id] = {
                "preferred_name": preferred_name,
                "definition": definition
            }

        self.logger.info(f"Extracted {len(data)} categorization classes")
        return data

    def extract_item_classes(self, root: et.Element) -> Dict[str, Dict[str, Optional[str]]]:
        """Extract item classes from the XML.

        Args:
            root: Root element of the XML tree

        Returns:
            Dictionary mapping class IDs to their attributes (preferred_name, definition, parent_id)
        """
        data = {}
        dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=self.ns)

        if dictionary_node is None:
            self.logger.warning("No dictionary node found")
            return data

        # Find all class elements with ITEM_CLASS_Type
        for elem in dictionary_node.findall(".//*[@xsi:type='ontoml:ITEM_CLASS_Type']",
                                            namespaces=self.ns):
            elem_id = elem.attrib.get("id")
            if elem_id is None:
                continue

            preferred_name = self._extract_text_field(elem, "preferred_name/label")
            definition = self._extract_text_field(elem, "definition/text")

            # Extract parent categorization
            subclass_elem = elem.find(".//subclass_of", namespaces=self.ns)
            parent_id = subclass_elem.get("idref") if subclass_elem is not None else None

            # Extract property references
            property_refs = elem.findall(".//property_ref", namespaces=self.ns)
            property_ids = [ref.get("idref") for ref in property_refs if ref.get("idref")]

            data[elem_id] = {
                "preferred_name": preferred_name,
                "definition": definition,
                "parent_id": parent_id,
                "property_ids": ",".join(property_ids) if property_ids else None
            }

        self.logger.info(f"Extracted {len(data)} item classes")
        return data

    def extract_properties(self, root: et.Element) -> Dict[str, Dict[str, Optional[str]]]:
        """Extract property definitions from the XML.

        Args:
            root: Root element of the XML tree

        Returns:
            Dictionary mapping property IDs to their attributes (preferred_name, definition, data_type, unit)
        """
        data = {}
        dictionary_node = root.find(".//ontoml:ontoml/dictionary", namespaces=self.ns)

        if dictionary_node is None:
            self.logger.warning("No dictionary node found")
            return data

        # Find all class elements with ITEM_CLASS_Type
        for elem in dictionary_node.findall(".//*[@xsi:type='ontoml:NON_DEPENDENT_P_DET_Type']",
                                                namespaces=self.ns):
            elem_id = elem.attrib.get("id")
            if elem_id is None:
                continue

            preferred_name = self._extract_text_field(elem, "preferred_name/label")
            definition = self._extract_text_field(elem, "definition/text")


            data[elem_id] = {
                "preferred_name": preferred_name,
                "definition": definition
            }

        self.logger.info(f"Extracted {len(data)} property classes")
        return data



def save_to_csv(data: Dict[str, Dict], output_path: str, logger: logging.Logger) -> None:
    """Save extracted data to CSV file.

    Args:
        data: Dictionary of extracted data
        output_path: Path where CSV should be saved
        logger: Logger instance for logging
    """
    if not data:
        logger.warning(f"No data to save to {output_path}")
        return

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame and save
    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "id"}, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV with {len(data)} records to: {output_path}")


def process_segment(
        segment: int,
        extractor: EClassExtractor,
        base_input_path: str,
        base_output_path: str
) -> Dict[str, Dict[str, Dict]]:
    """Process a single ECLASS segment and extract all data types.

    Args:
        segment: Segment number to process
        extractor: EClassExtractor instance
        base_input_path: Template path for input files (use {segment} placeholder)
        base_output_path: Template path for output files (use {segment} and {type} placeholders)

    Returns:
        Dictionary containing extracted data for all types
    """
    input_path = base_input_path.format(segment=segment)
    root = extractor.parse_xml(input_path)

    if root is None:
        return {"categorizations": {}, "item_classes": {}, "properties": {}}

    # Extract all data types
    categorizations = extractor.extract_categorizations(root)
    item_classes = extractor.extract_item_classes(root)
    properties = extractor.extract_properties(root)

    # Save to separate files
    save_to_csv(
        categorizations,
        base_output_path.format(segment=segment, type="categorizations"),
        extractor.logger
    )
    save_to_csv(
        item_classes,
        base_output_path.format(segment=segment, type="item-classes"),
        extractor.logger
    )
    save_to_csv(
        properties,
        base_output_path.format(segment=segment, type="properties"),
        extractor.logger
    )

    return {
        "categorizations": categorizations,
        "item_classes": item_classes,
        "properties": properties
    }


if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments
    segments = list(range(13, 52)) + [90]

    # Paths
    base_input_path = "../../data/original/ECLASS15_0_ADVANCED_EN_SG_{segment}.xml"
    base_output_path = "../../data/extracted/{type}/eclass-{segment}.csv"
    combined_output_path = "../../data/extracted/{type}/eclass-all.csv"

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    extractor = EClassExtractor(logger)

    # Storage for combined data
    all_data = {
        "categorizations": {},
        "item_classes": {},
        "properties": {}
    }

    # Process each segment
    for segment in segments:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue

        segment_data = process_segment(segment, extractor, base_input_path, base_output_path)

        # Accumulate data for combined files
        for data_type in all_data.keys():
            all_data[data_type].update(segment_data[data_type])

    # Save combined results for each data type
    for data_type, data in all_data.items():
        output_path = combined_output_path.format(type=data_type)
        save_to_csv(data, output_path, logger)
        logger.info(f"Completed extraction of {data_type}: {len(data)} total records")