export class VoiceSelector {
    constructor(voiceService) {
        this.voiceService = voiceService;
        this.elements = {
            voiceSearch: document.getElementById('voice-search'),
            voiceDropdown: document.getElementById('voice-dropdown'),
            voiceOptions: document.getElementById('voice-options'),
            selectedVoices: document.getElementById('selected-voices')
        };
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Voice search
        this.elements.voiceSearch.addEventListener('input', (e) => {
            const filteredVoices = this.voiceService.filterVoices(e.target.value);
            this.renderVoiceOptions(filteredVoices);
        });

        // Voice selection
        this.elements.voiceOptions.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox') {
                if (e.target.checked) {
                    this.voiceService.addVoice(e.target.value);
                } else {
                    this.voiceService.removeVoice(e.target.value);
                }
                this.updateSelectedVoicesDisplay();
            }
        });

        // Remove selected voice
        this.elements.selectedVoices.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-voice')) {
                const voice = e.target.dataset.voice;
                this.voiceService.removeVoice(voice);
                this.updateVoiceCheckbox(voice, false);
                this.updateSelectedVoicesDisplay();
            }
        });

        // Dropdown visibility
        this.elements.voiceSearch.addEventListener('focus', () => {
            this.elements.voiceDropdown.style.display = 'block';
            this.updateSearchPlaceholder();
        });

        document.addEventListener('click', (e) => {
            if (!this.elements.voiceSearch.contains(e.target) && 
                !this.elements.voiceDropdown.contains(e.target)) {
                this.elements.voiceDropdown.style.display = 'none';
            }
        });

        this.elements.voiceSearch.addEventListener('blur', () => {
            if (!this.elements.voiceSearch.value) {
                this.updateSearchPlaceholder();
            }
        });
    }

    renderVoiceOptions(voices) {
        this.elements.voiceOptions.innerHTML = voices
            .map(voice => `
                <label class="voice-option">
                    <input type="checkbox" value="${voice}" 
                        ${this.voiceService.getSelectedVoices().includes(voice) ? 'checked' : ''}>
                    ${voice}
                </label>
            `)
            .join('');
    }

    updateSelectedVoicesDisplay() {
        const selectedVoices = this.voiceService.getSelectedVoices();
        this.elements.selectedVoices.innerHTML = selectedVoices
            .map(voice => `
                <span class="selected-voice-tag">
                    ${voice}
                    <span class="remove-voice" data-voice="${voice}">×</span>
                </span>
            `)
            .join('');
        
        this.updateSearchPlaceholder();
    }

    updateSearchPlaceholder() {
        const hasSelected = this.voiceService.hasSelectedVoices();
        this.elements.voiceSearch.placeholder = hasSelected ? 
            'Search voices...' : 
            'Search and select voices...';
    }

    updateVoiceCheckbox(voice, checked) {
        const checkbox = this.elements.voiceOptions
            .querySelector(`input[value="${voice}"]`);
        if (checkbox) {
            checkbox.checked = checked;
        }
    }

    async initialize() {
        try {
            await this.voiceService.loadVoices();
            this.renderVoiceOptions(this.voiceService.getAvailableVoices());
            this.updateSelectedVoicesDisplay();
            return true;
        } catch (error) {
            console.error('Failed to initialize voice selector:', error);
            return false;
        }
    }
}

export default VoiceSelector;