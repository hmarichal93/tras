// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll <= 0) {
        navbar.style.boxShadow = 'none';
    } else {
        navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.3)';
    }
    
    lastScroll = currentScroll;
});

// Mobile menu toggle
const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (navToggle) {
    navToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        navToggle.classList.toggle('active');
    });
}

// Add fade-in animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.research-card, .project-card, .contact-card, .skill-category').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    observer.observe(el);
});

// Dynamic year in footer
const footer = document.querySelector('.footer p');
if (footer) {
    const currentYear = new Date().getFullYear();
    footer.innerHTML = footer.innerHTML.replace('2024', currentYear);
}

// Add keyboard navigation support
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        navLinks.classList.remove('active');
        navToggle.classList.remove('active');
    }
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
        hero.style.opacity = 1 - scrolled / 700;
    }
});

// Fetch latest release version from GitHub API
async function updateLatestRelease() {
    try {
        const response = await fetch('https://api.github.com/repos/hmarichal93/tras/releases/latest');
        if (!response.ok) {
            throw new Error('Failed to fetch latest release');
        }
        const release = await response.json();
        const version = release.tag_name.replace(/^v/, ''); // Remove 'v' prefix if present
        const tagName = release.tag_name;
        
        // Update version badge
        const versionText = document.getElementById('version-text');
        if (versionText) {
            versionText.textContent = `Version ${version}`;
        }
        
        // Update download button text
        const downloadText = document.getElementById('download-text');
        if (downloadText) {
            downloadText.textContent = `Download ${tagName}`;
        }
        
        // Update download link to point to latest release assets
        const downloadLink = document.getElementById('download-link');
        if (downloadLink) {
            // Find zip asset or use archive link
            const zipAsset = release.assets.find(asset => asset.name.endsWith('.zip'));
            if (zipAsset) {
                downloadLink.href = zipAsset.browser_download_url;
            } else {
                downloadLink.href = `https://github.com/hmarichal93/tras/archive/refs/tags/${tagName}.zip`;
            }
        }
        
        // Update installation download link
        const installDownloadLink = document.getElementById('install-download-link');
        if (installDownloadLink) {
            installDownloadLink.href = `https://github.com/hmarichal93/tras/archive/refs/tags/${tagName}.zip`;
            installDownloadLink.textContent = `Latest Release (${tagName})`;
        }
        
        // Update footer version
        const footerVersion = document.getElementById('footer-version');
        if (footerVersion) {
            footerVersion.textContent = tagName;
        }
        
        // Update meta description and title
        document.querySelector('meta[name="description"]').content = 
            `TRAS - Tree Ring Analyzer Suite ${tagName} - Professional dendrochronology software with automatic tree ring detection and measurement`;
        document.title = `TRAS - Tree Ring Analyzer Suite ${tagName}`;
        
    } catch (error) {
        console.error('Error fetching latest release:', error);
        // Fallback: keep "Latest" text or show error
        const versionText = document.getElementById('version-text');
        if (versionText) {
            versionText.textContent = 'Latest';
        }
        const downloadText = document.getElementById('download-text');
        if (downloadText) {
            downloadText.textContent = 'Download Latest';
        }
    }
}

// Call on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateLatestRelease);
} else {
    updateLatestRelease();
}

