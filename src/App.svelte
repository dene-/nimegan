<script lang="ts">
	import { SvelteToast, toast } from '@zerodevx/svelte-toast';
	import Blitz from 'blitz-resize';
	import { fade } from 'svelte/transition';

	// Components
	import Spinner from './components/spinner.svelte';

	let imgSrc = '';
	let finalSrc = '';
	let resizedImg;
	let files: FileList;
	let loading: boolean = false;

	$: if (files) {
		let reader = new FileReader();
		reader.readAsDataURL(files[0]);
		reader.onload = async () => {
			// Set preview image
			imgSrc = reader.result as string;

			const bAwait = Blitz.create();
			const maxSize = 1080;
			try {
				let output = await bAwait({
					source: reader.result,
					maxWidth: maxSize,
					maxHeight: maxSize,
					outputFormat: 'jpg',
					output: 'blob',
					quality: 0.95,
				});
				// data is a base 64 string you can attach to image src.
				resizedImg = output;
			} catch (err) {
				console.log(err);
			}
		};
	}

	function processPhoto() {
		if (!resizedImg || loading) return;

		let photoBlob = resizedImg;

		let formData = new FormData();
		formData.append('file', photoBlob, 'image.jpg');

		loading = true;

		fetch('http://terminaldogma.servebeer.com:1024', {
			method: 'post',
			body: formData,
			mode: 'cors',
		})
			.then((res) => res.blob())
			.then((blob) => {
				let url = URL.createObjectURL(blob);
				finalSrc = url;
				loading = false;
			})
			.catch((response) => {
				console.log(response);
				toast.push(response.error ? response.error : 'Error! D:');
				loading = false;
				finalSrc = resizedImg = imgSrc = files = undefined;
			});
	}
</script>

<header>
	<img src="logo.svg" alt="" id="logo" />
</header>

<main>
	<input type="file" accept="image/*" id="photoInput" name="photoInput" bind:files />
	<div id="photoWrapper">
		{#if imgSrc}
			{#if loading}
				<div transition:fade="{{ duration: 333 }}">
					<Spinner show="{loading}" />
				</div>
			{/if}
			{#if finalSrc}
				<img transition:fade="{{ duration: 333 }}" src="{finalSrc}" id="finalResult" alt="Result" />
			{/if}
			<img transition:fade="{{ duration: 333 }}" src="{imgSrc}" id="imageResult" alt="Result" class:blur="{loading}" />
		{:else}
			<button
				id="selectPhotoButton"
				on:click="{() => {
					document.getElementById('photoInput').click();
				}}"
			>
				Select Photo
			</button>
		{/if}
	</div>
</main>

<SvelteToast />

{#if imgSrc}
	<div id="actionButtons" class:gray="{loading}">
		<div id="buttonsWrapper">
			<div class="actionButton" on:click="{processPhoto}">
				<svg style="width: 30px; height: 30px" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="green" viewBox="0 0 24 24"><path d="M20.285 2l-11.285 11.567-5.286-5.011-3.714 3.716 9 8.728 15-15.285z"></path> </svg>
			</div>
			<div
				class="actionButton"
				on:click="{() => {
					if (!loading) {
						imgSrc = undefined;
						finalSrc = undefined;
					}
				}}"
			>
				<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="red" viewBox="0 0 24 24"><path d="M24 20.188l-8.315-8.209 8.2-8.282-3.697-3.697-8.212 8.318-8.31-8.203-3.666 3.666 8.321 8.24-8.206 8.313 3.666 3.666 8.237-8.318 8.285 8.203z"></path> </svg>
			</div>
		</div>
	</div>
{/if}

<style>
	:root {
		--toastBackground: rgb(65, 65, 65);
	}
	main {
		max-width: 100%;
		height: 100%;
		margin: 0 auto;

		display: flex;
		justify-content: center;
		flex-direction: column;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	header {
		position: fixed;
		background-color: #333;
		top: 0;

		height: 85px;
		width: 100%;
		text-align: center;
		padding: 15px 0;
	}

	#logo {
		height: 90%;
	}

	#photoInput {
		display: none;
	}

	#imageResult,
	#finalResult {
		position: absolute;
		max-width: 100%;
		max-height: 100%;
		border-radius: 40px;
		padding: 15px;
	}

	#finalResult {
		z-index: 2;
	}

	#photoWrapper {
		flex-grow: 1;
		display: flex;
		justify-content: center;
		align-items: center;

		transition: all 333ms linear;
	}

	#selectPhotoButton {
		border: 0;

		font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
		font-weight: bolder;
		text-transform: uppercase;

		width: 150px;
		height: 150px;
		border-radius: 100px;

		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.07), 0 4px 8px rgba(0, 0, 0, 0.07), 0 8px 16px rgba(0, 0, 0, 0.07), 0 16px 32px rgba(0, 0, 0, 0.07), 0 32px 64px rgba(0, 0, 0, 0.07);
	}

	.gray {
		filter: grayscale(100%);
	}

	.blur {
		filter: blur(5px);
	}

	#actionButtons {
		position: absolute;
		bottom: 0;
		width: 100%;
		height: 64px;
		display: flex;
		justify-content: center;
		flex-direction: row;

		padding: 15px 0;
		box-sizing: content-box;

		transition: all 333ms linear;
	}

	#buttonsWrapper {
		display: flex;
		flex-direction: row;
	}

	.actionButton {
		background-color: white;
		border-radius: 70px;

		height: 64px;
		width: 64px;

		display: flex;
		justify-content: center;
		align-items: center;
	}

	.actionButton:first-of-type {
		margin-right: 10px;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
