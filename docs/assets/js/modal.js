function openModal(imageSrc) {
    document.getElementById("modalImage").src = imageSrc;
    document.getElementById("imageModal").classList.add("is-active");
}

function closeModal() {
    document.getElementById("imageModal").classList.remove("is-active");
}
