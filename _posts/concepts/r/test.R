    # Create a sample student dataset
student_data <- data.frame(
  roll_number = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  name = c("Alice", "Bob", "Charlie", "David", "Eve", 
           "Frank", "Grace", "Henry", "Ivy", "Jack"),
  age = c(18, 19, 18, 20, 19, 18, 19, 20, 18, 19),
  marks = c(85, 92, 78, 95, 88, 82, 90, 87, 93, 85)
)

# Print the dataset
print(student_data)

# Load ggplot2 for better visualization
library(ggplot2)

# Create a bar plot for marks distribution
marks_plot <- ggplot(student_data, aes(x = roll_number, y = marks)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = marks), vjust = -0.5) +
  labs(title = "Student Marks Distribution",
       x = "Roll Number",
       y = "Marks") +
  theme_minimal()

# Create a scatter plot for age vs marks
age_marks_plot <- ggplot(student_data, aes(x = age, y = marks)) +
  geom_point(size = 3, color = "darkblue") +
  geom_text(aes(label = name), vjust = -1) +
  labs(title = "Age vs Marks Correlation",
       x = "Age",
       y = "Marks") +
  theme_minimal()

# Save the plots
ggsave("marks_distribution.png", marks_plot, width = 10, height = 6)
ggsave("age_marks_correlation.png", age_marks_plot, width = 10, height = 6)

# Print message
cat("Plots have been saved as 'marks_distribution.png' and 'age_marks_correlation.png' in the current directory.")
