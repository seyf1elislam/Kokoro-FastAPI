export class VoiceService {
    constructor() {
        this.availableVoices = [];
        this.selectedVoices = new Set();
    }

    async loadVoices() {
        try {
            const response = await fetch('/v1/audio/voices');
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to load voices');
            }
            
            const data = await response.json();
            if (!data.voices?.length) {
                throw new Error('No voices available');
            }

            this.availableVoices = data.voices;
            
            // Select first voice if none selected
            if (this.selectedVoices.size === 0) {
                const firstVoice = this.availableVoices.find(voice => voice && voice.trim());
                if (firstVoice) {
                    this.addVoice(firstVoice);
                }
            }

            return this.availableVoices;
        } catch (error) {
            console.error('Failed to load voices:', error);
            throw error;
        }
    }

    getAvailableVoices() {
        return this.availableVoices;
    }

    getSelectedVoices() {
        return Array.from(this.selectedVoices);
    }

    getSelectedVoiceString() {
        return Array.from(this.selectedVoices).join('+');
    }

    addVoice(voice) {
        if (this.availableVoices.includes(voice)) {
            this.selectedVoices.add(voice);
            return true;
        }
        return false;
    }

    removeVoice(voice) {
        return this.selectedVoices.delete(voice);
    }

    clearSelectedVoices() {
        this.selectedVoices.clear();
    }

    filterVoices(searchTerm) {
        if (!searchTerm) {
            return this.availableVoices;
        }
        
        const term = searchTerm.toLowerCase();
        return this.availableVoices.filter(voice => 
            voice.toLowerCase().includes(term)
        );
    }

    hasSelectedVoices() {
        return this.selectedVoices.size > 0;
    }
}

export default VoiceService;