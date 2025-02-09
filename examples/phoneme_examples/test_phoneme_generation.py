import requests
import os
import json

def main():
    # Test phoneme string
    phonemes = "hˈɛloʊ wˈɜrld"  # "Hello world" in phonemes
    
    try:
        print("\nTesting phoneme generation via API...")
        
        # Create request payload
        payload = {
            "phonemes": phonemes,
            "voice": "af_bella"  # Using bella voice
        }
        
        # Make request to the API endpoint
        response = requests.post(
            "http://localhost:8880/dev/generate_from_phonemes",
            json=payload,
            stream=True  # Enable streaming for audio data
        )
        
        # Check if request was successful
        if response.status_code == 200:
            # Create output directory if it doesn't exist
            os.makedirs("examples/phoneme_examples/output", exist_ok=True)
            
            # Save the audio response
            output_path = 'examples/phoneme_examples/output/phoneme_test.wav'
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"\nAudio saved to: {output_path}")
            print("\nPhoneme test completed successfully!")
            print(f"\nInput phonemes: {phonemes}")
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()