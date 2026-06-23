param(
    [string]$InstallRoot = "D:\AI\llama.cpp",
    [string]$LlamaCliPath = "",
    [ValidateSet("12", "13")]
    [string]$CudaMajor = "13",
    [string]$Tag = "latest",
    [string]$ModelPath = "D:\AI\Models\unsloth__gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf",
    [string]$MtpPath = "D:\AI\Models\unsloth__gemma-4-12b-it-GGUF\MTP\gemma-4-12b-it-Q8_0-MTP.gguf",
    [string]$Prompt = "Write a compact numbered list of practical tips for running local LLM inference. Use repeated phrasing: tip title, one sentence, tip title, one sentence. Continue until the answer is complete.",
    [int]$Context = 4096,
    [int]$MaxTokens = 256,
    [int]$SmokeTimeoutSeconds = 240,
    [int]$RunTimeoutSeconds = 1200,
    [int]$Runs = 3,
    [string]$GpuLayerCandidates = "auto,999,40,32,24,16,8,0,omit",
    [string]$MtpDraftGpuLayerCandidates = "999,auto,all,0,omit",
    [int]$SpecDraftMax = 4,
    [double]$Temperature = 0.2,
    [ValidateSet("1", "0", "true", "false", "on", "off", "auto", "omit")]
    [string]$FlashAttention = "on",
    [switch]$SkipDownload,
    [switch]$BaselineOnly,
    [switch]$MtpOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-JsonEvent {
    param(
        [hashtable]$Event,
        [string]$Path
    )
    $json = $Event | ConvertTo-Json -Compress -Depth 8
    Write-Host $json
    if ($Path) {
        Add-Content -LiteralPath $Path -Value $json
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Resolve-ReleaseTag {
    param([string]$RequestedTag)
    if ($RequestedTag -ne "latest") {
        return $RequestedTag
    }
    $resolved = gh release view --repo ggml-org/llama.cpp --json tagName --jq ".tagName"
    if ($LASTEXITCODE -ne 0 -or -not $resolved) {
        throw "Unable to resolve latest ggml-org/llama.cpp release tag with gh."
    }
    return $resolved.Trim()
}

function Select-ReleaseAsset {
    param(
        [string[]]$Assets,
        [string]$ResolvedTag,
        [string]$Cuda,
        [ValidateSet("bin", "dll")]
        [string]$Kind
    )

    if ($Kind -eq "bin") {
        $matches = $Assets | Where-Object {
            $_ -like "llama-$ResolvedTag-bin-win-cuda*$Cuda*x64.zip" -and
            $_ -notlike "cudart-*"
        }
    } else {
        $matches = $Assets | Where-Object {
            $_ -like "cudart-llama-bin-win-cuda*$Cuda*x64.zip"
        }
    }

    $matchList = @($matches)
    if ($matchList.Count -eq 0) {
        throw "Could not find $Kind asset for CUDA $Cuda in release $ResolvedTag. Assets: $($Assets -join ', ')"
    }

    return ($matchList | Sort-Object | Select-Object -Last 1)
}

function Ensure-LlamaRelease {
    param(
        [string]$Root,
        [string]$ResolvedTag,
        [string]$Cuda,
        [switch]$NoDownload
    )

    Require-Command gh
    $releaseRoot = Join-Path $Root $ResolvedTag
    $archiveRoot = Join-Path $releaseRoot "archives"
    $extractRoot = Join-Path $releaseRoot "native"
    New-Item -ItemType Directory -Force -Path $archiveRoot, $extractRoot | Out-Null

    $assetNames = gh release view $ResolvedTag --repo ggml-org/llama.cpp --json assets --jq ".assets[].name"
    if ($LASTEXITCODE -ne 0 -or -not $assetNames) {
        throw "Unable to list release assets for ggml-org/llama.cpp $ResolvedTag."
    }
    $assets = @($assetNames | Where-Object { $_ })
    $binAsset = Select-ReleaseAsset -Assets $assets -ResolvedTag $ResolvedTag -Cuda $Cuda -Kind bin
    $dllAsset = Select-ReleaseAsset -Assets $assets -ResolvedTag $ResolvedTag -Cuda $Cuda -Kind dll

    foreach ($asset in @($binAsset, $dllAsset)) {
        $zipPath = Join-Path $archiveRoot $asset
        if (-not (Test-Path -LiteralPath $zipPath)) {
            if ($NoDownload) {
                throw "Missing $zipPath and -SkipDownload was set."
            }
            gh release download $ResolvedTag --repo ggml-org/llama.cpp --pattern $asset --dir $archiveRoot --clobber
            if ($LASTEXITCODE -ne 0) {
                throw "Download failed for release asset $asset."
            }
        }
        Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force
    }

    $cli = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "llama-cli.exe" |
        Sort-Object FullName |
        Select-Object -First 1
    if (-not $cli) {
        throw "Could not find llama-cli.exe below $extractRoot."
    }

    $dllDirs = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "*.dll" |
        Select-Object -ExpandProperty DirectoryName -Unique
    $pathParts = @($cli.DirectoryName) + @($dllDirs)
    $env:PATH = (($pathParts | Select-Object -Unique) -join ";") + ";$env:PATH"

    return @{
        Cli = $cli.FullName
        ReleaseRoot = $releaseRoot
        BinAsset = $binAsset
        DllAsset = $dllAsset
    }
}

function Test-HelpFlag {
    param(
        [string]$HelpText,
        [string]$Flag
    )
    return $HelpText.Contains($Flag)
}

function Get-GpuArgs {
    param([string]$GpuLayers)
    if ($GpuLayers -eq "omit") {
        return @()
    }
    if ($GpuLayers -eq "all") {
        return @("--gpu-layers", "999")
    }
    return @("--gpu-layers", $GpuLayers)
}

function New-LlamaArgs {
    param(
        [string]$Mode,
        [string]$GpuLayers,
        [string]$DraftGpuLayers,
        [int]$TokenCount,
        [string]$HelpText
    )

    $args = @(
        "-m", $ModelPath,
        "-c", "$Context",
        "-n", "$TokenCount",
        "--temp", "$Temperature",
        "-p", $Prompt
    )
    $args += Get-GpuArgs -GpuLayers $GpuLayers
    if ($FlashAttention -ne "omit" -and (Test-HelpFlag -HelpText $HelpText -Flag "--flash-attn")) {
        $args += @("--flash-attn", $FlashAttention)
    }
    if (Test-HelpFlag -HelpText $HelpText -Flag "--single-turn") {
        $args += "--single-turn"
    }
    if (Test-HelpFlag -HelpText $HelpText -Flag "--simple-io") {
        $args += "--simple-io"
    }

    if ($Mode -eq "mtp") {
        $args += @(
            "--spec-type", "draft-mtp",
            "--spec-draft-model", $MtpPath,
            "--spec-draft-n-max", "$SpecDraftMax"
        )
        if ($DraftGpuLayers -ne "omit") {
            $args += @("--spec-draft-ngl", $DraftGpuLayers)
        }
    }

    return $args
}

function Invoke-NativeProcess {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [int]$TimeoutSeconds
    )

    $job = Start-Job -ScriptBlock {
        param(
            [string]$JobExecutable,
            [string[]]$JobArguments
        )

        & $JobExecutable @JobArguments 2>&1
        $jobExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        Write-Output "__LAAS_EXIT_CODE__=$jobExitCode"
    } -ArgumentList $Executable, $Arguments

    $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
    if (-not $completed) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        $partialOutput = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        return @{
            output = $partialOutput
            exit_code = -1
            timed_out = $true
        }
    }

    $received = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue

    $exitCode = 0
    $output = New-Object System.Collections.Generic.List[string]
    foreach ($line in $received) {
        $text = [string]$line
        if ($text.StartsWith("__LAAS_EXIT_CODE__=")) {
            $exitCode = [int]$text.Substring("__LAAS_EXIT_CODE__=".Length)
        } else {
            $output.Add($text)
        }
    }

    return @{
        output = @($output)
        exit_code = $exitCode
        timed_out = $false
    }
}

function Parse-LlamaPerf {
    param([string[]]$Output)
    $text = $Output -join "`n"
    $perf = @{
        tokens = $null
        tokens_per_second = $null
        prompt_tokens = $null
        prompt_tokens_per_second = $null
        acceptance = $null
    }

    $evalMatches = [regex]::Matches(
        $text,
        "eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(?:tokens|runs).*?,\s*([\d.]+)\s*tokens per second",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($evalMatches.Count -gt 0) {
        $m = $evalMatches[$evalMatches.Count - 1]
        $perf.tokens = [int]$m.Groups[1].Value
        $perf.tokens_per_second = [double]$m.Groups[2].Value
    }

    $promptMatches = [regex]::Matches(
        $text,
        "prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(?:tokens|runs).*?,\s*([\d.]+)\s*tokens per second",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($promptMatches.Count -gt 0) {
        $m = $promptMatches[$promptMatches.Count - 1]
        $perf.prompt_tokens = [int]$m.Groups[1].Value
        $perf.prompt_tokens_per_second = [double]$m.Groups[2].Value
    }

    $acceptanceMatches = [regex]::Matches(
        $text,
        "(?:acceptance|accepted).*?([\d.]+)\s*%",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($acceptanceMatches.Count -gt 0) {
        $perf.acceptance = [double]$acceptanceMatches[$acceptanceMatches.Count - 1].Groups[1].Value
    }

    $chatPerfMatches = [regex]::Matches(
        $text,
        "\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($chatPerfMatches.Count -gt 0) {
        $m = $chatPerfMatches[$chatPerfMatches.Count - 1]
        if ($null -eq $perf.prompt_tokens_per_second) {
            $perf.prompt_tokens_per_second = [double]$m.Groups[1].Value
        }
        if ($null -eq $perf.tokens_per_second) {
            $perf.tokens_per_second = [double]$m.Groups[2].Value
        }
    }

    return $perf
}

function Invoke-LlamaRun {
    param(
        [string]$Cli,
        [string]$Mode,
        [string]$GpuLayers,
        [string]$DraftGpuLayers = "omit",
        [int]$Run,
        [int]$TokenCount,
        [string]$HelpText,
        [string]$LogRoot,
        [string]$JsonlPath,
        [int]$TimeoutSeconds
    )

    $args = New-LlamaArgs -Mode $Mode -GpuLayers $GpuLayers -DraftGpuLayers $DraftGpuLayers -TokenCount $TokenCount -HelpText $HelpText
    $logMode = if ($Mode -eq "mtp") {
        "mtp-draft-{0}" -f ($DraftGpuLayers -replace '[^A-Za-z0-9_.-]', '_')
    } else {
        $Mode
    }
    $logPath = Join-Path $LogRoot ("{0}-run-{1}.log" -f $logMode, $Run)
    Write-JsonEvent -Event @{
        event = "native_benchmark_run_start"
        mode = $Mode
        run = $Run
        gpu_layers = $GpuLayers
        draft_gpu_layers = $DraftGpuLayers
        timeout_seconds = $TimeoutSeconds
        log_path = $logPath
    } -Path $JsonlPath

    $started = Get-Date
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $processResult = Invoke-NativeProcess -Executable $Cli -Arguments $args -TimeoutSeconds $TimeoutSeconds
    $output = @($processResult["output"])
    $exitCode = [int]$processResult["exit_code"]
    $timedOut = [bool]$processResult["timed_out"]
    $sw.Stop()
    $output | Set-Content -LiteralPath $logPath
    $perf = Parse-LlamaPerf -Output $output

    $event = @{
        event = "native_benchmark_run"
        mode = $Mode
        run = $Run
        exit_code = $exitCode
        timed_out = $timedOut
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        started_at = $started.ToString("o")
        gpu_layers = $GpuLayers
        draft_gpu_layers = $DraftGpuLayers
        tokens = $perf.tokens
        tokens_per_second = $perf.tokens_per_second
        prompt_tokens = $perf.prompt_tokens
        prompt_tokens_per_second = $perf.prompt_tokens_per_second
        acceptance = $perf.acceptance
        log_path = $logPath
    }
    Write-JsonEvent -Event $event -Path $JsonlPath

    if ($timedOut) {
        throw "llama-cli timed out after $TimeoutSeconds seconds for mode=$Mode run=$Run gpu_layers=$GpuLayers. See $logPath."
    }
    if ($exitCode -ne 0) {
        throw "llama-cli failed for mode=$Mode run=$Run gpu_layers=$GpuLayers. See $logPath."
    }

    return $event
}

function Select-WorkingGpuLayers {
    param(
        [string]$Cli,
        [string]$HelpText,
        [string]$LogRoot,
        [string]$JsonlPath
    )

    foreach ($candidate in ($GpuLayerCandidates -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        try {
            Write-JsonEvent -Event @{
                event = "gpu_candidate_start"
                gpu_layers = $candidate
            } -Path $JsonlPath
            [void](Invoke-LlamaRun -Cli $Cli -Mode "baseline-smoke" -GpuLayers $candidate -Run 1 -TokenCount 16 -HelpText $HelpText -LogRoot $LogRoot -JsonlPath $JsonlPath -TimeoutSeconds $SmokeTimeoutSeconds)
            Write-JsonEvent -Event @{
                event = "gpu_candidate_selected"
                gpu_layers = $candidate
            } -Path $JsonlPath
            return $candidate
        } catch {
            Write-JsonEvent -Event @{
                event = "gpu_candidate_failed"
                gpu_layers = $candidate
                error = $_.Exception.Message
            } -Path $JsonlPath
        }
    }

    throw "No GPU-layer candidate worked. Tried: $GpuLayerCandidates"
}

function Write-BenchmarkSummary {
    param(
        [string]$Mode,
        [object[]]$Events,
        [string]$GpuLayers,
        [string]$DraftGpuLayers,
        [string]$JsonlPath
    )

    $eventList = @($Events)
    $tokenRates = @($eventList | Where-Object { $null -ne $_["tokens_per_second"] } | ForEach-Object { [double]$_["tokens_per_second"] })
    if ($tokenRates.Count -gt 0) {
        Write-JsonEvent -Event @{
            event = "native_benchmark_summary"
            mode = $Mode
            runs = $eventList.Count
            gpu_layers = $GpuLayers
            draft_gpu_layers = $DraftGpuLayers
            avg_tokens_per_second = [math]::Round(($tokenRates | Measure-Object -Average).Average, 3)
            best_tokens_per_second = [math]::Round(($tokenRates | Measure-Object -Maximum).Maximum, 3)
            min_tokens_per_second = [math]::Round(($tokenRates | Measure-Object -Minimum).Minimum, 3)
        } -Path $JsonlPath
    } else {
        Write-JsonEvent -Event @{
            event = "native_benchmark_summary_unparsed"
            mode = $Mode
            runs = $eventList.Count
            gpu_layers = $GpuLayers
            draft_gpu_layers = $DraftGpuLayers
            note = "No generation throughput line was parsed. Inspect the run logs."
        } -Path $JsonlPath
    }
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Main model not found: $ModelPath"
}
if (-not (Test-Path -LiteralPath $MtpPath)) {
    throw "MTP draft model not found: $MtpPath"
}

if ($LlamaCliPath) {
    if (-not (Test-Path -LiteralPath $LlamaCliPath)) {
        throw "llama-cli not found: $LlamaCliPath"
    }
    $resolvedTag = $Tag
    $release = @{
        Cli = (Resolve-Path -LiteralPath $LlamaCliPath).Path
        ReleaseRoot = Split-Path -Parent (Resolve-Path -LiteralPath $LlamaCliPath).Path
        BinAsset = "existing"
        DllAsset = "existing"
    }
} else {
    Require-Command gh
    $resolvedTag = Resolve-ReleaseTag -RequestedTag $Tag
    $release = Ensure-LlamaRelease -Root $InstallRoot -ResolvedTag $resolvedTag -Cuda $CudaMajor -NoDownload:$SkipDownload
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$benchRoot = Join-Path $InstallRoot "benchmarks\gemma4-12b-mtp-$timestamp"
New-Item -ItemType Directory -Force -Path $benchRoot | Out-Null
$jsonlPath = Join-Path $benchRoot "results.jsonl"

$helpText = (& $release.Cli --help 2>&1) -join "`n"

Write-JsonEvent -Event @{
    event = "native_benchmark_start"
    tag = $resolvedTag
    cuda_major = $CudaMajor
    cli = $release.Cli
    bin_asset = $release.BinAsset
    dll_asset = $release.DllAsset
    model_path = $ModelPath
    mtp_path = $MtpPath
    context = $Context
    max_tokens = $MaxTokens
    smoke_timeout_seconds = $SmokeTimeoutSeconds
    run_timeout_seconds = $RunTimeoutSeconds
    runs = $Runs
    spec_draft_max = $SpecDraftMax
    mtp_draft_gpu_layer_candidates = $MtpDraftGpuLayerCandidates
    results = $jsonlPath
} -Path $jsonlPath

$selectedGpuLayers = Select-WorkingGpuLayers -Cli $release.Cli -HelpText $helpText -LogRoot $benchRoot -JsonlPath $jsonlPath

if (-not $MtpOnly) {
    $events = @()
    for ($i = 1; $i -le $Runs; $i++) {
        $events += Invoke-LlamaRun -Cli $release.Cli -Mode "baseline" -GpuLayers $selectedGpuLayers -Run $i -TokenCount $MaxTokens -HelpText $helpText -LogRoot $benchRoot -JsonlPath $jsonlPath -TimeoutSeconds $RunTimeoutSeconds
    }
    Write-BenchmarkSummary -Mode "baseline" -Events $events -GpuLayers $selectedGpuLayers -DraftGpuLayers "omit" -JsonlPath $jsonlPath
}

if (-not $BaselineOnly) {
    $mtpSucceeded = $false
    foreach ($draftCandidate in ($MtpDraftGpuLayerCandidates -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $events = @()
        try {
            Write-JsonEvent -Event @{
                event = "mtp_candidate_start"
                gpu_layers = $selectedGpuLayers
                draft_gpu_layers = $draftCandidate
            } -Path $jsonlPath
            for ($i = 1; $i -le $Runs; $i++) {
                $events += Invoke-LlamaRun -Cli $release.Cli -Mode "mtp" -GpuLayers $selectedGpuLayers -DraftGpuLayers $draftCandidate -Run $i -TokenCount $MaxTokens -HelpText $helpText -LogRoot $benchRoot -JsonlPath $jsonlPath -TimeoutSeconds $RunTimeoutSeconds
            }
            Write-BenchmarkSummary -Mode "mtp" -Events $events -GpuLayers $selectedGpuLayers -DraftGpuLayers $draftCandidate -JsonlPath $jsonlPath
            Write-JsonEvent -Event @{
                event = "mtp_candidate_selected"
                gpu_layers = $selectedGpuLayers
                draft_gpu_layers = $draftCandidate
            } -Path $jsonlPath
            $mtpSucceeded = $true
            break
        } catch {
            Write-JsonEvent -Event @{
                event = "mtp_candidate_failed"
                gpu_layers = $selectedGpuLayers
                draft_gpu_layers = $draftCandidate
                completed_runs = @($events).Count
                error = $_.Exception.Message
            } -Path $jsonlPath
        }
    }
    if (-not $mtpSucceeded) {
        Write-JsonEvent -Event @{
            event = "native_benchmark_mtp_unavailable"
            gpu_layers = $selectedGpuLayers
            draft_gpu_layer_candidates = $MtpDraftGpuLayerCandidates
            note = "Native llama.cpp could run the main Gemma 4 12B model, but every draft-mtp candidate failed or timed out."
        } -Path $jsonlPath
    }
}

Write-JsonEvent -Event @{
    event = "native_benchmark_done"
    results = $jsonlPath
    log_dir = $benchRoot
} -Path $jsonlPath
