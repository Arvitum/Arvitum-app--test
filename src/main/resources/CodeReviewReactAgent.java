package com.epam.codereview.service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.stereotype.Service;
import org.springframework.util.StreamUtils;

import com.epam.codereview.config.CodeReviewProperties;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class CodeReviewReactAgent {

  private final ChatModel chatModel;
  private final ChatOptions chatOptions;
  private final ToolCallingManager toolCallingManager;
  private final CodeReviewProperties codeReviewProperties;
  private final PullRequestReferenceParser pullRequestReferenceParser;

  private String loadSystemPrompt() {
    try {
      return StreamUtils.copyToString(codeReviewProperties.getSystemPrompt().getInputStream(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to load system prompt from: " + codeReviewProperties.getSystemPrompt(), e);
    }
  }

  public String interact(String userInput) {
    return interact(pullRequestReferenceParser.parse(userInput));
  }

  public String interact(PullRequestRef pullRequestRef) {
    String systemPrompt = loadSystemPrompt();
    PullRequestReviewActivity activity = new PullRequestReviewActivity();

    List<Message> messages = new ArrayList<>();
    messages.add(new SystemMessage(systemPrompt));
    messages.add(new UserMessage(buildUserMessage(pullRequestRef)));

    Prompt prompt = new Prompt(messages, chatOptions);
    log.info("Starting PR review owner={} repo={} pr={}",
      pullRequestRef.owner(), pullRequestRef.repo(), pullRequestRef.prNumber());

    try {
      ChatResponse chatResponse = requestModelResponse(prompt, pullRequestRef, activity);
      log.debug("Initial model response received owner={} repo={} pr={} hasToolCalls={}",
        pullRequestRef.owner(), pullRequestRef.repo(), pullRequestRef.prNumber(), chatResponse.hasToolCalls());

      while (chatResponse.hasToolCalls()) {
        activity.recordToolRound(extractToolCalls(chatResponse));
        log.info("PR review tool round={} owner={} repo={} pr={} tools={}",
          activity.toolRounds(), pullRequestRef.owner(), pullRequestRef.repo(), pullRequestRef.prNumber(), activity.latestRoundTools());

        ToolExecutionResult toolExecutionResult = toolCallingManager.executeToolCalls(prompt, chatResponse);
        List<Message> conversationHistory = new ArrayList<>(toolExecutionResult.conversationHistory());
        activity.recordToolExecutionResult(conversationHistory);
        if (activity.shouldStopAfterSuccessfulFinalReview(conversationHistory)) {
          verifyWorkflow(pullRequestRef, activity);
          String review = buildExecutionSummary(
            pullRequestRef,
            activity,
            "Pending-review comment tool was unavailable; stopped after final review submission.");
          log.info(
            "Completed PR review owner={} repo={} pr={} toolRounds={} inlineCommentsPosted={} finalReviewSubmitted={}",
            pullRequestRef.owner(),
            pullRequestRef.repo(),
            pullRequestRef.prNumber(),
            activity.toolRounds(),
            activity.inlineCommentsPosted(),
            activity.finalReviewSubmitted());
          return review;
        }
        conversationHistory.add(new SystemMessage(buildRuntimeReminder(activity)));
        prompt = new Prompt(conversationHistory, chatOptions);
        chatResponse = requestModelResponse(prompt, pullRequestRef, activity);
        log.debug("Model response after tool execution owner={} repo={} pr={} hasToolCalls={}",
          pullRequestRef.owner(), pullRequestRef.repo(), pullRequestRef.prNumber(), chatResponse.hasToolCalls());
      }

      verifyWorkflow(pullRequestRef, activity);

      String review = chatResponse.getResult().getOutput().getText();
      log.info(
        "Completed PR review owner={} repo={} pr={} toolRounds={} inlineCommentsPosted={} finalReviewSubmitted={}",
        pullRequestRef.owner(),
        pullRequestRef.repo(),
        pullRequestRef.prNumber(),
        activity.toolRounds(),
        activity.inlineCommentsPosted(),
        activity.finalReviewSubmitted());
      return review;
    } catch (RuntimeException exception) {
      log.error("PR review execution failed owner={} repo={} pr={}: {}",
        pullRequestRef.owner(), pullRequestRef.repo(), pullRequestRef.prNumber(), exception.getMessage(), exception);
      throw exception;
    }
  }

  private String buildExecutionSummary(
    PullRequestRef pullRequestRef, PullRequestReviewActivity activity, String note) {
    return """
      Reviewed %s PR #%d. Inline comments posted: %d. Final PR review submitted: %s. %s
      """.formatted(
      pullRequestRef.repository(),
      pullRequestRef.prNumber(),
      activity.inlineCommentsPosted(),
      activity.finalReviewSubmitted() ? "yes" : "no",
      note);
  }

  private String buildUserMessage(PullRequestRef pullRequestRef) {
    return """
      Review this GitHub pull request using the required PR review workflow.

      Normalized PR target:
      - owner: %s
      - repo: %s
      - repositorySlug: %s
      - pullRequestNumber: %d
      - pullRequestUrl: %s

      GitHub tool argument rules:
      - use `owner` exactly as provided
      - use `repo` exactly as provided
      - never pass `owner/repo` as the repo parameter
      - if GitHub MCP tool names differ from prompt examples, use the runtime tool names that match the required action

      Return only the concise execution summary after the required GitHub review actions are completed.
      """.formatted(
      pullRequestRef.owner(),
      pullRequestRef.repo(),
      pullRequestRef.repository(),
      pullRequestRef.prNumber(),
      pullRequestRef.pullRequestUrl());
  }

  private List<AssistantMessage.ToolCall> extractToolCalls(ChatResponse chatResponse) {
    AssistantMessage assistantMessage = chatResponse.getResult().getOutput();
    if (assistantMessage == null || assistantMessage.getToolCalls() == null) {
      return List.of();
    }
    return assistantMessage.getToolCalls();
  }

  private ChatResponse requestModelResponse(
    Prompt prompt, PullRequestRef pullRequestRef, PullRequestReviewActivity activity) {
    Prompt currentPrompt = prompt;

    for (int attempt = 0; attempt < 3; attempt++) {
      ToolCallNormalizationResult normalizationResult =
        normalizeToolCalls(chatModel.call(currentPrompt), pullRequestRef, activity);
      ChatResponse chatResponse = normalizationResult.chatResponse();

      if (!normalizationResult.hasSkippedToolCalls()
        || chatResponse.hasToolCalls()
        || hasText(chatResponse)) {
        return chatResponse;
      }

      List<Message> instructions = new ArrayList<>(currentPrompt.getInstructions());
      instructions.add(new SystemMessage(buildToolCorrection(normalizationResult)));
      currentPrompt = new Prompt(instructions, chatOptions);
    }

    throw new PullRequestReviewExecutionException("The agent kept requesting redundant or unavailable pull request review tools.");
  }

  private ToolCallNormalizationResult normalizeToolCalls(
    ChatResponse chatResponse, PullRequestRef pullRequestRef, PullRequestReviewActivity activity) {
    AssistantMessage assistantMessage = chatResponse.getResult().getOutput();
    if (assistantMessage == null || assistantMessage.getToolCalls() == null || assistantMessage.getToolCalls().isEmpty()) {
      return new ToolCallNormalizationResult(chatResponse, false, false, false);
    }

    List<AssistantMessage.ToolCall> deduplicatedToolCalls = deduplicateToolCalls(assistantMessage.getToolCalls());
    boolean skippedPostFinalReviewToolCalls = false;
    if (activity.finalReviewSubmitted()) {
      skippedPostFinalReviewToolCalls = !deduplicatedToolCalls.isEmpty();
      deduplicatedToolCalls = List.of();
    }
    boolean skippedRepeatedInvalidFileContentCalls = false;
    if (activity.shouldSkipFurtherFileContentRequests()) {
      List<AssistantMessage.ToolCall> filteredToolCalls = deduplicatedToolCalls.stream()
        .filter(toolCall -> !activity.shouldSkipFileContentTool(toolCall.name()))
        .toList();
      skippedRepeatedInvalidFileContentCalls = filteredToolCalls.size() != deduplicatedToolCalls.size();
      deduplicatedToolCalls = filteredToolCalls;
    }
    boolean skippedUnavailablePendingReviewCommentCalls = false;
    if (activity.pendingReviewUnavailable()) {
      List<AssistantMessage.ToolCall> filteredToolCalls = deduplicatedToolCalls.stream()
        .filter(toolCall -> !activity.shouldSkipPendingReviewCommentTool(toolCall.name()))
        .toList();
      skippedUnavailablePendingReviewCommentCalls = filteredToolCalls.size() != deduplicatedToolCalls.size();
      deduplicatedToolCalls = filteredToolCalls;
    }

    if (deduplicatedToolCalls.size() == assistantMessage.getToolCalls().size()
      && !skippedPostFinalReviewToolCalls
      && !skippedRepeatedInvalidFileContentCalls
      && !skippedUnavailablePendingReviewCommentCalls) {
      return new ToolCallNormalizationResult(chatResponse, false, false, false);
    }

    log.info("Normalized tool calls owner={} repo={} pr={} original={} normalized={}",
      pullRequestRef.owner(),
      pullRequestRef.repo(),
      pullRequestRef.prNumber(),
      assistantMessage.getToolCalls().stream().map(AssistantMessage.ToolCall::name).toList(),
      deduplicatedToolCalls.stream().map(AssistantMessage.ToolCall::name).toList());

    AssistantMessage normalizedAssistantMessage = AssistantMessage.builder()
      .content(assistantMessage.getText())
      .properties(assistantMessage.getMetadata())
      .toolCalls(deduplicatedToolCalls)
      .media(assistantMessage.getMedia())
      .build();

    Generation normalizedGeneration = new Generation(normalizedAssistantMessage, chatResponse.getResult().getMetadata());
    return new ToolCallNormalizationResult(
      new ChatResponse(List.of(normalizedGeneration), chatResponse.getMetadata()),
      skippedPostFinalReviewToolCalls,
      skippedUnavailablePendingReviewCommentCalls,
      skippedRepeatedInvalidFileContentCalls);
  }

  private List<AssistantMessage.ToolCall> deduplicateToolCalls(List<AssistantMessage.ToolCall> toolCalls) {
    Set<ToolCallKey> seen = new LinkedHashSet<>();
    List<AssistantMessage.ToolCall> deduplicated = new ArrayList<>();
    for (AssistantMessage.ToolCall toolCall : toolCalls) {
      ToolCallKey key = new ToolCallKey(toolCall.type(), toolCall.name(), toolCall.arguments());
      if (seen.add(key)) {
        deduplicated.add(toolCall);
      }
    }
    return deduplicated;
  }

  private String buildRuntimeReminder(PullRequestReviewActivity activity) {
    StringBuilder reminder = new StringBuilder("""
      Runtime reminder:
      - At most one inline review comment is allowed for the entire PR.
      - As soon as you have one valid inline comment, post it and then submit the final PR review immediately.
      - Do not continue reviewing the PR after the first inline comment is posted.
      - If one available PR read tool already returned the metadata and changed files or diff, do not call that PR read tool a second time.
      """);

    if (activity.requestedPullRequestMetadata()) {
      reminder.append("""

        - Pull request metadata has already been read. Do not call the PR metadata tool again unless the previous call failed.
        """);
    }
    if (activity.listedPullRequestFiles()) {
      reminder.append("""

        - Changed files or diff data have already been retrieved. Do not call the same PR read or diff tool again unless the previous call failed.
        - Your next step is to choose one exact changed source file path from the existing diff data, then call `get_file_contents`, `retrieveCodeLanguage`, and `retrieveCodeConvention`.
        """);
    }
    if (activity.inlineCommentsPosted() > 0 && !activity.finalReviewSubmitted()) {
      reminder.append("""

        - The single allowed inline comment has already been posted. Your only remaining action is to submit the final PR review summary.
        """);
    }
    if (activity.finalReviewSubmitted()) {
      reminder.append("""

        - The final pull request review has already been submitted successfully.
        - Do not call any more tools.
        - Return the concise execution summary text now.
        """);
    }
    if (activity.pendingReviewUnavailable()) {
      reminder.append("""

        - `add_comment_to_pending_review` already failed because there is no pending review.
        - Make sure you called `pull_request_review_write` with `method: "create"` (without `event`) first to open a pending review before calling `add_comment_to_pending_review`.
        - If the pending review creation already failed, fall back to `pull_request_review_write` with `method: "create"`, `event: "COMMENT"`, and describe the finding in `body`.
        """);
    }
    if (activity.invalidFilePathRequested()) {
      reminder.append("""

        - A `get_file_contents` call already failed because the requested path was not found in the repository.
        - Use only exact file paths returned by the pull request diff or changed-files data.
        - Do not guess alternate file paths or retry the same file with path variants.
        """);
    }

    return reminder.toString();
  }

  private String buildToolCorrection(ToolCallNormalizationResult normalizationResult) {
    StringBuilder correction = new StringBuilder("Runtime correction:\n");
    if (normalizationResult.skippedPostFinalReviewToolCalls()) {
      correction.append("""
        - The final pull request review has already been submitted successfully.
        - Do not call any more tools.
        - Return the concise execution summary text now.
        """);
    }
    if (normalizationResult.skippedUnavailablePendingReviewCommentCalls()) {
      correction.append("""
        - `add_comment_to_pending_review` failed because no pending review exists.
        - Do not call `add_comment_to_pending_review` again.
        - Call `pull_request_review_write` with `method: "create"` first to create the pending review, then retry `add_comment_to_pending_review`.
        - If creating a pending review is not possible, fall back to `pull_request_review_write` with `method: "create"`, `event: "COMMENT"`, and describe the finding in `body`.
        """);
    }
    if (normalizationResult.skippedRepeatedInvalidFileContentCalls()) {
      correction.append("""
        - Recent `get_file_contents` requests failed repeatedly because the requested paths were not valid repository files.
        - Do not call `get_file_contents` again in this review.
        - Use the existing diff context and any file content already retrieved.
        - If you already have one valid finding on a changed added line, submit it with `pull_request_review_write`.
        - Otherwise submit the final PR review summary now and stop.
        """);
    }
    return correction.toString();
  }

  private boolean hasText(ChatResponse chatResponse) {
    AssistantMessage assistantMessage = chatResponse.getResult().getOutput();
    return assistantMessage != null && assistantMessage.getText() != null && !assistantMessage.getText().isBlank();
  }

  private void verifyWorkflow(PullRequestRef pullRequestRef, PullRequestReviewActivity activity) {
    if (!activity.hasPullRequestToolActivity()) {
      throw new PullRequestReviewExecutionException("The agent finished without executing pull request review tools.");
    }
    if (!activity.requestedPullRequestMetadata()) {
      throw new PullRequestReviewExecutionException("The agent finished without reading pull request metadata.");
    }
    if (!activity.listedPullRequestFiles()) {
      throw new PullRequestReviewExecutionException("The agent finished without retrieving changed pull request files or diff.");
    }
    if (!activity.finalReviewSubmitted()) {
      throw new PullRequestReviewExecutionException("The agent finished without submitting a final pull request review.");
    }

    log.info("Verified PR review workflow owner={} repo={} pr={} inlineCommentsPosted={} finalReviewSubmitted={}",
      pullRequestRef.owner(),
      pullRequestRef.repo(),
      pullRequestRef.prNumber(),
      activity.inlineCommentsPosted(),
      activity.finalReviewSubmitted());
  }

  private static final class PullRequestReviewActivity {

    private static final Set<String> METADATA_TOOLS = Set.of(
      "get_pull_request",
      "pull_request_read");
    private static final Set<String> FILE_OR_DIFF_TOOLS = Set.of(
      "list_pull_request_files",
      "get_pull_request_diff",
      "pull_request_read");
    private static final Set<String> INLINE_COMMENT_TOOLS = Set.of(
      "add_pull_request_review_comment",
      "create_pull_request_review_comment",
      "add_comment_to_pending_review",
      "pull_request_review_comment_write");
    private static final Set<String> FINAL_REVIEW_TOOLS = Set.of(
      "create_pull_request_review",
      "pull_request_review_write",
      "submit_pull_request_review");

    private final Set<String> seenTools = new HashSet<>();
    private int toolRounds;
    private int inlineCommentsPosted;
    private boolean finalReviewSubmitted;
    private boolean pendingReviewUnavailable;
    private boolean invalidFilePathRequested;
    private int invalidFilePathFailures;
    private List<String> latestRoundTools = List.of();

    void recordToolRound(List<AssistantMessage.ToolCall> toolCalls) {
      List<String> toolNames = toolCalls.stream().map(AssistantMessage.ToolCall::name).toList();
      toolRounds++;
      latestRoundTools = toolNames;
      seenTools.addAll(toolNames);
      inlineCommentsPosted += (int) toolNames.stream()
        .filter(PullRequestReviewActivity::isInlineCommentTool)
        .count();
      finalReviewSubmitted = finalReviewSubmitted || toolCalls.stream()
        .anyMatch(tc -> isFinalReviewTool(tc.name()) && isReviewSubmission(tc.arguments()));
    }

    private static boolean isReviewSubmission(String arguments) {
      return arguments != null && (
        arguments.contains("\"submit_pending\"") ||
        arguments.contains("\"event\"")
      );
    }

    boolean hasPullRequestToolActivity() {
      return seenTools.stream().anyMatch(PullRequestReviewActivity::isPullRequestTool);
    }

    boolean requestedPullRequestMetadata() {
      return seenTools.stream().anyMatch(PullRequestReviewActivity::isMetadataTool);
    }

    boolean listedPullRequestFiles() {
      return seenTools.stream().anyMatch(PullRequestReviewActivity::isFileOrDiffTool);
    }

    void recordToolExecutionResult(List<Message> conversationHistory) {
      String conversationText = conversationHistory.stream()
        .map(Message::getText)
        .filter(text -> text != null && !text.isBlank())
        .reduce("", (left, right) -> left + "\n" + right);
      pendingReviewUnavailable = pendingReviewUnavailable
        || conversationText.contains("No pending review found for the viewer")
        || conversationText.contains("is not pending");
      boolean failedToGetFileContents =
        conversationText.contains("Failed to get file contents. The path does not point to a file or directory");
      invalidFilePathRequested = invalidFilePathRequested || failedToGetFileContents;
      if (failedToGetFileContents) {
        invalidFilePathFailures++;
      }
    }

    boolean pendingReviewUnavailable() {
      return pendingReviewUnavailable;
    }

    boolean invalidFilePathRequested() {
      return invalidFilePathRequested;
    }

    boolean shouldSkipFurtherFileContentRequests() {
      return invalidFilePathFailures >= 2;
    }

    boolean shouldSkipPendingReviewCommentTool(String toolName) {
      return "add_comment_to_pending_review".equals(toolName);
    }

    boolean shouldSkipFileContentTool(String toolName) {
      return "get_file_contents".equals(toolName);
    }

    boolean shouldStopAfterSuccessfulFinalReview(List<Message> conversationHistory) {
      if (!pendingReviewUnavailable || !finalReviewSubmitted || !latestRoundTools.stream().anyMatch(PullRequestReviewActivity::isFinalReviewTool)) {
        return false;
      }
      if (conversationHistory.isEmpty()) {
        return true;
      }
      String lastMessageText = conversationHistory.getLast().getText();
      return !lastMessageText.contains("Error calling tool")
        && !lastMessageText.contains("No pending review found");
    }

    boolean finalReviewSubmitted() {
      return finalReviewSubmitted;
    }

    int inlineCommentsPosted() {
      return inlineCommentsPosted;
    }

    int toolRounds() {
      return toolRounds;
    }

    List<String> latestRoundTools() {
      return latestRoundTools;
    }

    private static boolean isPullRequestTool(String toolName) {
      return isMetadataTool(toolName)
        || isFileOrDiffTool(toolName)
        || isInlineCommentTool(toolName)
        || isFinalReviewTool(toolName);
    }

    private static boolean isMetadataTool(String toolName) {
      return METADATA_TOOLS.contains(toolName);
    }

    private static boolean isFileOrDiffTool(String toolName) {
      return FILE_OR_DIFF_TOOLS.contains(toolName)
        || toolName.contains("pull_request_files")
        || toolName.contains("pull_request_diff");
    }

    private static boolean isInlineCommentTool(String toolName) {
      return INLINE_COMMENT_TOOLS.contains(toolName)
        || (toolName.contains("review_comment") && toolName.contains("write"));
    }

    private static boolean isFinalReviewTool(String toolName) {
      return FINAL_REVIEW_TOOLS.contains(toolName)
        || (toolName.contains("pull_request_review")
        && !toolName.contains("comment")
        && (toolName.contains("write") || toolName.contains("create") || toolName.contains("submit")));
    }
  }

  private record ToolCallKey(String type, String name, String arguments) {
  }

  private record ToolCallNormalizationResult(
    ChatResponse chatResponse,
    boolean skippedPostFinalReviewToolCalls,
    boolean skippedUnavailablePendingReviewCommentCalls,
    boolean skippedRepeatedInvalidFileContentCalls) {

    boolean hasSkippedToolCalls() {
      return skippedPostFinalReviewToolCalls
        || skippedUnavailablePendingReviewCommentCalls
        || skippedRepeatedInvalidFileContentCalls;
    }
  }
}
