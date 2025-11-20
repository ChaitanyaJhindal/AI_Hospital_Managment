"""
RAG-based Summarizer using Groq API
Converts technical results from A* and CSP into simple English explanations
"""

import os
from groq import Groq

class RAGSummarizer:
    def __init__(self, api_key=None):
        """Initialize Groq client with API key"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "your-api-key-here")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Using available Groq model
    
    def create_bed_allocation_summary(self, allocation_data, total_patients):
        """
        Generate simple English summary of A* bed allocation results
        
        Args:
            allocation_data: List of bed allocation dictionaries
            total_patients: Total number of patients
        
        Returns:
            str: Simple English summary
        """
        # Prepare context from allocation data
        context = f"""
You are a hospital administrator assistant. Explain the following bed allocation results in simple, easy-to-understand English.

ALLOCATION RESULTS:
- Total Patients: {total_patients}
- Beds Allocated: {len(allocation_data)}

Sample Allocations:
"""
        for alloc in allocation_data[:5]:  # First 5 for context
            context += f"\n- Patient {alloc.get('Patient', 'Unknown')} (Severity: {alloc.get('Severity', 0):.2f}) → {alloc.get('Assigned_Bed', 'N/A')} (Distance: {alloc.get('Distance_Cost', 'N/A')})"
        
        context += """

Please provide:
1. A brief overview of what was done
2. How patients were prioritized
3. What the distance cost means
4. Any important insights

Keep the explanation simple, as if explaining to someone without technical knowledge.
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful hospital assistant who explains medical resource allocation in simple, clear language. Avoid technical jargon."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def create_schedule_summary(self, schedule_data):
        """
        Generate simple English summary of CSP scheduling results
        
        Args:
            schedule_data: List of scheduling dictionaries
        
        Returns:
            str: Simple English summary
        """
        if not schedule_data or (len(schedule_data) == 1 and "Error" in schedule_data[0]):
            return "⚠️ No feasible schedule could be created with the current constraints."
        
        # Prepare context from schedule data
        context = f"""
You are a hospital administrator assistant. Explain the following surgery scheduling results in simple, easy-to-understand English.

SCHEDULING RESULTS:
- Total Surgeries Scheduled: {len(schedule_data)}

Surgery Schedule:
"""
        for sched in schedule_data:
            context += f"\n- Patient {sched.get('Patient_ID', 'Unknown')} (Severity: {sched.get('Severity', 0):.2f}) → {sched.get('Doctor', 'N/A')} in {sched.get('Room', 'N/A')} at {sched.get('Time', 'N/A')}"
        
        context += """

Please provide:
1. A brief overview of the surgery schedule
2. How the scheduling was optimized
3. Key constraints that were satisfied (no double-booking, etc.)
4. Any important insights about the schedule

Keep the explanation simple and clear.
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful hospital assistant who explains surgery scheduling in simple, clear language. Avoid technical jargon."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def create_combined_summary(self, allocation_data, schedule_data, total_patients):
        """
        Generate a comprehensive summary of both A* and CSP results
        
        Args:
            allocation_data: List of bed allocation dictionaries
            schedule_data: List of scheduling dictionaries
            total_patients: Total number of patients
        
        Returns:
            str: Comprehensive simple English summary
        """
        # Prepare comprehensive context
        context = f"""
You are a hospital administrator assistant. Provide a comprehensive summary of the hospital resource management system's results.

BED ALLOCATION (A* Algorithm):
- Total Patients: {total_patients}
- Beds Allocated: {len(allocation_data)}
- Top 3 High-Severity Patients:
"""
        sorted_allocs = sorted(allocation_data, key=lambda x: x.get('Severity', 0), reverse=True)
        for alloc in sorted_allocs[:3]:
            context += f"\n  • Patient {alloc.get('Patient', 'Unknown')} - Severity {alloc.get('Severity', 0):.2f} → {alloc.get('Assigned_Bed', 'N/A')}"
        
        context += f"""

SURGERY SCHEDULING (CSP):
- Surgeries Scheduled: {len(schedule_data) if schedule_data and 'Error' not in schedule_data[0] else 0}
"""
        if schedule_data and 'Error' not in schedule_data[0]:
            context += "- Schedule:\n"
            for sched in schedule_data[:5]:
                context += f"  • {sched.get('Patient_ID', 'Unknown')} with {sched.get('Doctor', 'N/A')} at {sched.get('Time', 'N/A')} in {sched.get('Room', 'N/A')}\n"
        
        context += """

Please provide a comprehensive yet simple summary that:
1. Explains what the system accomplished
2. Highlights how critical patients were prioritized
3. Explains the scheduling optimization
4. Provides actionable insights for hospital staff
5. Uses simple language that anyone can understand

Format the response with clear sections and bullet points.
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful hospital administrator assistant who explains complex hospital management results in simple, actionable language. Use clear sections and bullet points."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=800
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# Standalone function for easy import
def summarize_results(allocation_data=None, schedule_data=None, total_patients=0):
    """
    Convenience function to generate summaries
    
    Args:
        allocation_data: Bed allocation results
        schedule_data: Surgery schedule results
        total_patients: Total number of patients
    
    Returns:
        str: Summary text
    """
    summarizer = RAGSummarizer()
    
    if allocation_data and schedule_data:
        return summarizer.create_combined_summary(allocation_data, schedule_data, total_patients)
    elif allocation_data:
        return summarizer.create_bed_allocation_summary(allocation_data, total_patients)
    elif schedule_data:
        return summarizer.create_schedule_summary(schedule_data)
    else:
        return "No data provided for summary."
