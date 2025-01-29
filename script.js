// Handle Colorful Box Selection for File Source
document.getElementById('deviceBox').addEventListener('click', function() {
    showUploadArea();
});
document.getElementById('gdriveBox').addEventListener('click', function() {
    loadPicker(); // Call Google Drive picker
});
document.getElementById('onedriveBox').addEventListener('click', function() {
    openOneDrivePicker(); // Call OneDrive picker
});
document.getElementById('dropboxBox').addEventListener('click', function() {
    openDropboxPicker(); // Call Dropbox picker
});
document.getElementById('urlBox').addEventListener('click', function() {
    showUrlInput();
});

// Show file input for device upload
function showUploadArea() {
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('urlInput').style.display = 'none';
}

// Show URL input for URL uploads
function showUrlInput() {
    document.getElementById('urlInput').style.display = 'block';
    document.getElementById('uploadArea').style.display = 'none';
}

// URL Upload Logic
document.getElementById('submitUrlBtn').addEventListener('click', function() {
    const url = document.getElementById('urlField').value;
    if (url) {
        alert('URL Submitted: ' + url);
    } else {
        alert('Please enter a valid URL');
    }
})
    // Function to handle URL entry
    function enterURL() {
        document.getElementById('urlInput').style.display = 'block';
    }

    // Handle URL submission
    document.getElementById('submitUrlBtn').addEventListener('click', function() {
        const url = document.getElementById('urlField').value;
        if (url) {
            alert('URL submitted: ' + url);
            // Hide the input section after submission
            document.getElementById('urlInput').style.display = 'none';
        } else {
            alert('Please enter a valid URL');
        }
    });