import os
import openai
from typing import Dict, Any

class TenderAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def analyze_tender(self, text: str) -> Dict[str, Any]:
        try:
            max_text_length = 30000
            if len(text) > max_text_length:
                text = text[:max_text_length]

            prompt = self._create_analysis_prompt(text)

            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": "You are an expert in analyzing tender documents."},
                          {"role": "user", "content": prompt}],
                max_tokens=4000
            )

            analysis_text = response.choices[0].message.content
            structured_data = self._parse_analysis(analysis_text)

            return {
                "full_analysis": analysis_text,
                "structured_data": structured_data
            }

        except Exception as e:
            raise Exception(f"Error analyzing tender: {str(e)}")

    def _create_analysis_prompt(self, text: str) -> str:
        return f"""Analyze this tender document comprehensively and extract structured information.

        Your response should be well-structured with the following sections:

        # TENDER SUMMARY
        Provide a concise summary of the tender opportunity.

        # BASIC INFORMATION
        - Tender Reference Number:
        - Issuing Organization:
        - Submission Deadline:
        - Project Location:
        - Estimated Budget:

        # KEY REQUIREMENTS
        List the main technical, financial, and operational requirements.

        # ELIGIBILITY CRITERIA
        List the mandatory criteria that bidders must meet.

        # EVALUATION CRITERIA
        Explain how bids will be evaluated and scored.

        # REQUIRED DOCUMENTS
        List all documents that must be submitted.

        # COMPLIANCE CHECKLIST
        Create a checklist of critical compliance points.

        # WINNING STRATEGY
        Provide strategic recommendations to increase chances of success.

        # RISKS AND MITIGATIONS
        Identify potential risks and suggest mitigation strategies.

        # Tender Text:
        {text}
        """

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        sections = {}
        current_section = None
        current_content = []

        for line in analysis_text.split("\n"):
            if line.startswith("# "):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                    current_content = []
                current_section = line[2:].strip()
            elif current_section:
                current_content.append(line)

        if current_section and current_content:
            sections[current_section] = "\n".join(current_content)

        return sections
